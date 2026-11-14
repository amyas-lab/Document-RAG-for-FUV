"""FastAPI backend for the Applied Mathematics handbook assistant.

Run locally:   uvicorn api.main:app --reload --port 8000     (from source_code/)
Swagger UI:    http://localhost:8000/docs
Web UI:        http://localhost:8000/ui/   (local dev; in Docker the UI is served by nginx)
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator
from functools import lru_cache
from typing import Literal

import pymupdf
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.chain import RAGPipeline
from rag.config import ROOT, load_config
from rag.index import build_index, index_status
from rag.loaders import find_documents

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

CFG = load_config()
EMBEDDING_FAMILIES = ("embed", "bge", "minilm")   # names of embedding-only Ollama models

app = FastAPI(
    title="FUV Applied Mathematics Handbook Assistant",
    version="1.0.0",
    description="Local RAG over the B.S. Applied Mathematics Major Handbook (Ollama + Qdrant).",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_lock = threading.Lock()
_pipeline: RAGPipeline | None = None


def pipeline() -> RAGPipeline:
    """Created lazily so the API starts even while Ollama is still pulling models."""
    global _pipeline
    with _lock:
        if _pipeline is None:
            try:
                _pipeline = RAGPipeline(CFG)
            except Exception as e:  # Ollama / Qdrant not reachable yet
                log.exception("Pipeline initialisation failed")
                raise HTTPException(503, f"RAG pipeline not ready: {e}") from e
        return _pipeline


# --------------------------------------------------------------------------- schemas
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What are the requirements for the Applied Mathematics minor?"])
    history: list[Message] = Field(default_factory=list)
    model: str | None = Field(None, examples=[CFG.llm_model])
    top_k: int | None = Field(None, ge=1, le=20)
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    mode: Literal["strict", "fallback"] | None = None


class Source(BaseModel):
    page: int | None
    pages: list[int | None]
    section: str
    score: float
    ocr: bool
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    standalone_query: str
    model: str
    latency_ms: dict[str, int]


class IndexRequest(BaseModel):
    force: bool = False


def _kwargs(req: ChatRequest) -> dict:
    return dict(history=[m.model_dump() for m in req.history], model=req.model,
                top_k=req.top_k, temperature=req.temperature, mode=req.mode)


# --------------------------------------------------------------------------- endpoints
@app.get("/health")
def health() -> dict:
    ollama_ok = qdrant_ok = False
    status: dict = {}
    try:
        ollama_ok = requests.get(f"{CFG.ollama_url}/api/tags", timeout=3).ok
    except requests.RequestException:
        pass
    try:
        status = index_status(CFG)
        qdrant_ok = True
    except Exception:
        log.warning("Qdrant not reachable", exc_info=True)
    return {
        "status": "ok" if ollama_ok and qdrant_ok and status.get("exists") else "degraded",
        "ollama": ollama_ok,
        "qdrant": qdrant_ok,
        "qdrant_mode": "server" if CFG.qdrant_url else "embedded",
        "collection": status.get("collection", CFG.collection_name),
        "n_chunks": status.get("points", 0),
        "index_up_to_date": status.get("up_to_date", False),
        "defaults": {"model": CFG.llm_model, "embed_model": CFG.embed_model, "top_k": CFG.top_k,
                     "temperature": CFG.temperature, "mode": CFG.prompt_mode,
                     "parser": CFG.pdf_parser, "chunking": CFG.chunking, "retrieval": CFG.retrieval_mode},
    }


@app.get("/models")
def list_models() -> dict:
    """Installed Ollama chat models (embedding models are filtered out)."""
    try:
        tags = requests.get(f"{CFG.ollama_url}/api/tags", timeout=5).json().get("models", [])
    except requests.RequestException as e:
        raise HTTPException(503, f"Ollama not reachable: {e}") from e
    names = sorted(m["name"] for m in tags if not any(f in m["name"] for f in EMBEDDING_FAMILIES))
    return {"default": CFG.llm_model, "models": names}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> dict:
    try:
        return pipeline().answer(req.question, **_kwargs(req))
    except HTTPException:
        raise
    except Exception as e:
        log.exception("chat failed")
        raise HTTPException(500, str(e)) from e


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Server-Sent Events: one `sources` event, many `token` events, then `done` (or `error`)."""
    rag = pipeline()

    def events() -> Iterator[str]:
        try:
            for kind, payload in rag.stream(req.question, **_kwargs(req)):
                yield _sse(kind, payload)
        except Exception as e:
            log.exception("stream failed")
            yield _sse("error", {"detail": str(e)})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/index")
def reindex(req: IndexRequest | None = None) -> dict:
    """Re-index data/documents (only if documents or settings changed, unless force=true)."""
    global _pipeline
    with _lock:
        try:
            result = build_index(CFG, force=bool(req and req.force))
        except Exception as e:
            log.exception("indexing failed")
            raise HTTPException(500, str(e)) from e
        _pipeline = None   # rebuild the store handle on the next request
    return result


def _document(name: str):
    """Resolve a document by file name; only files found in data/documents are served."""
    for path in find_documents(CFG.data_path):
        if path.name == name:
            return path
    raise HTTPException(404, f"Unknown document: {name}")


@lru_cache(maxsize=16)
def _page_count(path: str, mtime: float) -> int:
    with pymupdf.open(path) as doc:
        return doc.page_count


@app.get("/documents")
def list_documents() -> dict:
    """Indexed source files with their page counts (used by the UI to show a cited page)."""
    docs = []
    for path in find_documents(CFG.data_path):
        pages = _page_count(str(path), path.stat().st_mtime) if path.suffix.lower() == ".pdf" else 0
        docs.append({"name": path.name, "pages": pages})
    return {"documents": docs}


@lru_cache(maxsize=64)
def _render_page(path: str, mtime: float, page: int, width: int) -> bytes:
    with pymupdf.open(path) as doc:
        pg = doc[page - 1]
        zoom = width / pg.rect.width
        return pg.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False).tobytes("png")


@app.get("/documents/{name}/pages/{page}.png")
def get_page_image(name: str, page: int, width: int = Query(900, ge=200, le=2000)) -> Response:
    """One PDF page rendered as PNG, so the UI can show the cited page in any browser."""
    path = _document(name)
    if path.suffix.lower() != ".pdf":
        raise HTTPException(400, "Only PDF documents have pages")
    mtime = path.stat().st_mtime
    if not 1 <= page <= _page_count(str(path), mtime):
        raise HTTPException(404, f"Page {page} out of range")
    png = _render_page(str(path), mtime, page, width)
    return Response(png, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/documents/{name}")
def get_document(name: str) -> FileResponse:
    """Serve an indexed document inline. Only files found in data/documents are served."""
    path = _document(name)
    media = "application/pdf" if path.suffix.lower() == ".pdf" else "text/plain; charset=utf-8"
    return FileResponse(path, media_type=media, headers={"Content-Disposition": f'inline; filename="{name}"'})


WEB_DIR = ROOT / "web"
if WEB_DIR.is_dir():   # local development: serve the static UI from the API process
    app.mount("/ui", StaticFiles(directory=WEB_DIR, html=True), name="ui")

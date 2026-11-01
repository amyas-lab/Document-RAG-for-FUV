"""Vector index in Qdrant (dense embeddings from Ollama + sparse BM25 for hybrid search).

* Server mode (Docker): ``QDRANT_URL=http://qdrant:6333``
* Embedded mode (local dev, no server): ``QDRANT_PATH=data/qdrant_local``

Every point stores a ``corpus_hash`` (files + parser/chunking settings). On startup the
index is rebuilt automatically when the documents or the processing settings change.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import uuid
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

from .chunking import chunk_documents
from .config import Config
from .loaders import file_sha256, find_documents, load_documents

log = logging.getLogger(__name__)

DENSE, SPARSE = "dense", "sparse"

# Task-specific prefixes recommended by the embedding model authors: (query, document)
EMBED_PREFIXES = {
    "nomic-embed-text": ("search_query: ", "search_document: "),
    "mxbai-embed-large": ("Represent this sentence for searching relevant passages: ", ""),
}


class PrefixedOllamaEmbeddings(Embeddings):
    """OllamaEmbeddings + the query/document prefixes some models were trained with."""

    def __init__(self, model: str, base_url: str):
        self._inner = OllamaEmbeddings(model=model, base_url=base_url)
        self.query_prefix, self.doc_prefix = EMBED_PREFIXES.get(model.split(":")[0], ("", ""))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(self.query_prefix + text)


# --------------------------------------------------------------------------- clients
_clients: dict[tuple[str, str], QdrantClient] = {}


def get_client(url: str, path: str) -> QdrantClient:
    # Embedded mode takes a file lock, so a process must reuse one client instance.
    key = (url, path)
    if key not in _clients:
        _clients[key] = QdrantClient(url=url) if url else QdrantClient(path=path)
    return _clients[key]


@atexit.register
def _close_clients():
    # Close clients explicitly; otherwise qdrant-client warns during interpreter shutdown.
    for client in _clients.values():
        try:
            client.close()
        except Exception:
            pass


def client_for(cfg: Config) -> QdrantClient:
    return get_client(cfg.qdrant_url, "" if cfg.qdrant_url else str(cfg.path(cfg.qdrant_path)))


@lru_cache(maxsize=4)
def get_embeddings(model: str, base_url: str) -> PrefixedOllamaEmbeddings:
    return PrefixedOllamaEmbeddings(model, base_url)


@lru_cache(maxsize=1)
def get_sparse() -> FastEmbedSparse:
    return FastEmbedSparse(model_name="Qdrant/bm25")


# --------------------------------------------------------------------------- index state
def corpus_hash(cfg: Config) -> str:
    """Fingerprint of the documents and every setting that changes the chunks."""
    state = {
        "files": [(p.name, file_sha256(p)) for p in find_documents(cfg.data_path)],
        "settings": [cfg.pdf_parser, cfg.ocr_enabled, cfg.ocr_dpi, cfg.ocr_min_chars,
                     cfg.chunking, cfg.chunk_size, cfg.chunk_overlap, cfg.embed_model],
    }
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:16]


def _stored_hash(client: QdrantClient, collection: str) -> str | None:
    if not client.collection_exists(collection):
        return None
    points, _ = client.scroll(collection, limit=1, with_payload=True, with_vectors=False)
    return points[0].payload.get("metadata", {}).get("corpus_hash") if points else None


def index_status(cfg: Config) -> dict:
    client = client_for(cfg)
    name = cfg.collection_name
    exists = client.collection_exists(name)
    return {
        "collection": name,
        "exists": exists,
        "points": client.count(name).count if exists else 0,
        "up_to_date": exists and _stored_hash(client, name) == corpus_hash(cfg),
    }


# --------------------------------------------------------------------------- build / load
def build_index(cfg: Config, force: bool = False) -> dict:
    """(Re)build the collection if it is missing, stale, or ``force`` is set."""
    client = client_for(cfg)
    name = cfg.collection_name
    current = corpus_hash(cfg)
    if not force and _stored_hash(client, name) == current:
        n = client.count(name).count
        log.info("Index '%s' is up to date (%d chunks).", name, n)
        return {"collection": name, "chunks": n, "rebuilt": False}

    log.info("Building index '%s' ...", name)
    pages = load_documents(cfg)
    chunks = chunk_documents(pages, cfg)
    for c in chunks:
        c.metadata["corpus_hash"] = current
    log.info("  %d pages -> %d chunks (parser=%s, chunking=%s)", len(pages), len(chunks), cfg.pdf_parser, cfg.chunking)

    embeddings = get_embeddings(cfg.embed_model, cfg.ollama_url)
    dim = len(embeddings.embed_query("dimension probe"))
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        name,
        vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    store = _store(cfg, client, RetrievalMode.HYBRID)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{name}/{i}")) for i in range(len(chunks))]
    store.add_documents(chunks, ids=ids, batch_size=32)
    log.info("  indexed %d chunks into Qdrant (%s mode, dim=%d)", len(chunks),
             "server" if cfg.qdrant_url else "embedded", dim)
    return {"collection": name, "chunks": len(chunks), "rebuilt": True}


def _store(cfg: Config, client: QdrantClient, mode: RetrievalMode) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client,
        collection_name=cfg.collection_name,
        embedding=get_embeddings(cfg.embed_model, cfg.ollama_url),
        sparse_embedding=get_sparse() if mode != RetrievalMode.DENSE else None,
        retrieval_mode=mode,
        vector_name=DENSE,
        sparse_vector_name=SPARSE,
    )


def get_store(cfg: Config, ensure_index: bool = True) -> QdrantVectorStore:
    """Vector store for querying; builds the index first if needed."""
    if ensure_index:
        build_index(cfg)
    mode = RetrievalMode.HYBRID if cfg.retrieval_mode == "hybrid" else RetrievalMode.DENSE
    return _store(cfg, client_for(cfg), mode)

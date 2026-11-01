"""RAG chain: (history-aware) query -> retrieve -> prompt -> LLM -> answer + sources."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from . import prompts
from .config import Config
from .index import get_store

log = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.store = get_store(cfg)

    # ------------------------------------------------------------------ pieces
    def llm(self, model: str | None = None, temperature: float | None = None) -> ChatOllama:
        return ChatOllama(
            model=model or self.cfg.llm_model,
            base_url=self.cfg.ollama_url,
            temperature=self.cfg.temperature if temperature is None else temperature,
            num_ctx=self.cfg.num_ctx,
        )

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        results = self.store.similarity_search_with_score(query, k=k or self.cfg.top_k)
        return [
            {
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "pages": d.metadata.get("pages") or [d.metadata.get("page")],
                "section": d.metadata.get("section", ""),
                "ocr": d.metadata.get("ocr", False),
                "score": round(float(score), 4),
                "text": d.page_content,
            }
            for d, score in results
        ]

    def standalone_query(self, question: str, history: list[dict] | None, model: str | None = None) -> str:
        """Turn a follow-up ("what are its prerequisites?") into a self-contained query."""
        if not history:
            return question
        messages = [
            SystemMessage(prompts.REWRITE_SYSTEM),
            HumanMessage(f"CONVERSATION:\n{prompts.format_history(history)}\n\nLATEST MESSAGE: {question}"),
        ]
        rewritten = self.llm(model, temperature=0).invoke(messages).content.strip().strip('"')
        return rewritten or question

    def build_messages(self, question: str, sources: list[dict], mode: str):
        context = prompts.format_context(sources)
        if mode == "fallback":
            return [HumanMessage(prompts.FALLBACK_TEMPLATE.format(context=context, question=question))]
        return [SystemMessage(prompts.STRICT_SYSTEM),
                HumanMessage(prompts.STRICT_USER.format(context=context, question=question))]

    # ------------------------------------------------------------------ API
    def _prepare(self, question, history, model, top_k, mode):
        t0 = time.perf_counter()
        query = self.standalone_query(question, history, model)
        sources = self.retrieve(query, top_k)
        t1 = time.perf_counter()
        messages = self.build_messages(query, sources, mode or self.cfg.prompt_mode)
        return query, sources, messages, t0, t1

    def answer(self, question: str, history: list[dict] | None = None, *, model: str | None = None,
               top_k: int | None = None, temperature: float | None = None, mode: str | None = None) -> dict:
        query, sources, messages, t0, t1 = self._prepare(question, history, model, top_k, mode)
        answer = prompts.strip_dangling_refusal(self.llm(model, temperature).invoke(messages).content.strip())
        t2 = time.perf_counter()
        return {
            "answer": answer,
            "sources": sources,
            "standalone_query": query,
            "model": model or self.cfg.llm_model,
            "latency_ms": {"retrieval": round((t1 - t0) * 1000), "generation": round((t2 - t1) * 1000)},
        }

    def stream(self, question: str, history: list[dict] | None = None, *, model: str | None = None,
               top_k: int | None = None, temperature: float | None = None,
               mode: str | None = None) -> Iterator[tuple[str, object]]:
        """Yields ("sources", [...]) first, then ("token", str)..., then ("done", result)."""
        query, sources, messages, t0, t1 = self._prepare(question, history, model, top_k, mode)
        yield "sources", {"sources": sources, "standalone_query": query}
        parts = []
        for chunk in self.llm(model, temperature).stream(messages):
            if chunk.content:
                parts.append(chunk.content)
                yield "token", chunk.content
        t2 = time.perf_counter()
        yield "done", {
            "answer": prompts.strip_dangling_refusal("".join(parts).strip()),
            "sources": sources,
            "standalone_query": query,
            "model": model or self.cfg.llm_model,
            "latency_ms": {"retrieval": round((t1 - t0) * 1000), "generation": round((t2 - t1) * 1000)},
        }

"""Command-line interface.

    python -m rag.cli extract              # dump extracted page text (inspect PDF parsing)
    python -m rag.cli index [--reindex]    # build / refresh the Qdrant index
    python -m rag.cli chunks --page 21     # inspect chunks
    python -m rag.cli ask -q "..."         # ask a question
    python -m rag.cli chat                 # interactive multi-turn chat in the terminal

Global options (--parser, --chunking, --model, ...) override config.yaml for one run.
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap

from .config import load_config


def _cfg(args):
    return load_config().with_overrides(
        pdf_parser=args.parser, chunking=args.chunking, llm_model=args.model,
        embed_model=args.embed, top_k=args.k, prompt_mode=args.mode, data_dir=args.data_dir,
        retrieval_mode=args.retrieval,
    )


def cmd_extract(args):
    from .loaders import load_documents

    cfg = _cfg(args)
    out_dir = cfg.path("data/extracted")
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = load_documents(cfg)
    by_source: dict[str, list] = {}
    for d in docs:
        by_source.setdefault(d.metadata["source"], []).append(d)
    for source, pages in by_source.items():
        out = out_dir / f"{source.rsplit('.', 1)[0]}.{cfg.pdf_parser}.md"
        out.write_text("\n\n".join(
            f"<!-- page {d.metadata['page']}{' (OCR)' if d.metadata['ocr'] else ''} -->\n{d.page_content}" for d in pages))
        chars = sum(len(d.page_content) for d in pages)
        print(f"{source}: {len(pages)} pages, {chars:,} characters -> {out}")


def cmd_index(args):
    from .index import build_index

    stats = build_index(_cfg(args), force=args.reindex)
    print(f"Collection '{stats['collection']}': {stats['chunks']} chunks"
          f" ({'rebuilt' if stats['rebuilt'] else 'already up to date'})")


def cmd_chunks(args):
    from .chunking import chunk_documents
    from .loaders import load_documents

    cfg = _cfg(args)
    chunks = chunk_documents(load_documents(cfg), cfg)
    print(f"{len(chunks)} chunks (parser={cfg.pdf_parser}, chunking={cfg.chunking})\n")
    for c in chunks:
        if args.page and args.page not in c.metadata.get("pages", [c.metadata["page"]]):
            continue
        if args.grep and args.grep.lower() not in c.page_content.lower():
            continue
        m = c.metadata
        print(f"--- chunk {m['chunk_index']} | page {m['page']} | {len(c.page_content)} chars | {m.get('section', '')}")
        print(c.page_content if args.full else textwrap.shorten(c.page_content, 300))
        print()


def _print_result(result: dict, show_context: bool):
    print(f"\nA: {result['answer']}\n")
    print("Sources:")
    for i, s in enumerate(result["sources"], 1):
        print(f"  [{i}] {s['source']} p.{s['page']}  score={s['score']}  {s['section'][:80]}")
        if show_context:
            print(textwrap.indent(s["text"], "      | "))
    lat = result["latency_ms"]
    print(f"\n({result['model']} | retrieval {lat['retrieval']} ms, generation {lat['generation']} ms)")


def cmd_ask(args):
    from .chain import RAGPipeline

    rag = RAGPipeline(_cfg(args))
    print(f"Q: {args.question}")
    _print_result(rag.answer(args.question), args.show_context)


def cmd_chat(args):
    from .chain import RAGPipeline

    rag = RAGPipeline(_cfg(args))
    history: list[dict] = []
    print("Applied Mathematics handbook chat. Type 'exit' to quit, 'reset' to clear history.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if q.lower() == "reset":
            history = []
            continue
        if not q:
            continue
        result = rag.answer(q, history)
        if result["standalone_query"] != q:
            print(f"  (searched for: {result['standalone_query']})")
        _print_result(result, args.show_context)
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": result["answer"]}]


def main(argv=None):
    p = argparse.ArgumentParser(prog="rag", description="Local RAG over the FUV Applied Mathematics handbook")
    p.add_argument("--parser", choices=["pypdf", "pymupdf"])
    p.add_argument("--chunking", choices=["fixed", "structure"])
    p.add_argument("--model", help="Ollama LLM, e.g. qwen2.5:3b")
    p.add_argument("--embed", help="Ollama embedding model, e.g. nomic-embed-text")
    p.add_argument("--retrieval", choices=["hybrid", "dense"])
    p.add_argument("--mode", choices=["strict", "fallback"], help="prompt mode")
    p.add_argument("-k", type=int, help="top-K chunks")
    p.add_argument("--data-dir")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("extract", help="dump extracted text per page").set_defaults(func=cmd_extract)
    s = sub.add_parser("index", help="build or refresh the vector index")
    s.add_argument("--reindex", action="store_true", help="force a full rebuild")
    s.set_defaults(func=cmd_index)
    s = sub.add_parser("chunks", help="inspect chunks")
    s.add_argument("--page", type=int)
    s.add_argument("--grep")
    s.add_argument("--full", action="store_true")
    s.set_defaults(func=cmd_chunks)
    s = sub.add_parser("ask", help="ask one question")
    s.add_argument("-q", "--question", default="What are the requirements for a minor in Applied Mathematics?")
    s.add_argument("--show-context", action="store_true")
    s.set_defaults(func=cmd_ask)
    s = sub.add_parser("chat", help="interactive multi-turn chat")
    s.add_argument("--show-context", action="store_true")
    s.set_defaults(func=cmd_chat)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose or args.command == "index" else logging.WARNING,
                        format="%(message)s")
    for noisy in ("httpx", "httpcore", "fastembed"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())

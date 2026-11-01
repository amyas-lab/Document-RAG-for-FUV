"""Evaluate retrieval and answers on the handbook test set.

    # 1) Retrieval-only ablation (no LLM): parser x chunking x retrieval mode x embedding model
    python -m eval.run_eval retrieval

    # 2) End-to-end answers for several LLMs (strict prompt by default)
    python -m eval.run_eval answers --models qwen2.5:0.5b,qwen2.5:1.5b,qwen2.5:3b,llama3.2:3b
    python -m eval.run_eval answers --models qwen2.5:3b --mode fallback

    # 3) Aggregate every results file into eval/results/summary.md
    python -m eval.run_eval report

Answers are labelled automatically from the key facts in questions.yaml:
    correct        all key facts present
    incomplete     some key facts present
    incorrect      no key fact present
    false_refusal  said "not in the handbook" although the answer is there
    correct_refusal / hallucination   for out-of-scope questions
Automatic labels are a first pass. Corrections from the manual review live in manual_labels.yaml\nand are applied by the report; the error analysis is in analysis.md.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from rag.config import load_config

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REFUSAL = re.compile(
    r"couldn.t find|could not find|cannot find|can.t find|not (mentioned|provided|available|specified|stated|found|included|listed)"
    r"|does not (contain|mention|provide|specify|include)|doesn.t (contain|mention|provide|specify|include)|no information",
    re.I,
)


# --------------------------------------------------------------------------- data
def load_questions(path: Path = HERE / "questions.yaml", ids: list[str] | None = None) -> list[dict]:
    qs = yaml.safe_load(path.read_text())["questions"]
    return [q for q in qs if not ids or q["id"] in ids]


def page_metrics(sources: list[dict], gold_pages: list[int], facts: list[str] | None = None, ks=(1, 3, 5)) -> dict:
    """Retrieval metrics.

    Page level: Hit@k, MRR and gold-page recall (every page a chunk spans counts).
    Evidence level: context_recall = share of the question's key facts that appear in the
    retrieved chunks. Page hits can be lucky (another chunk on the same page); context recall
    tells whether the evidence the LLM needs was actually retrieved."""
    if not gold_pages:
        return {}
    gold = set(gold_pages)
    hits = [bool(gold & set(s["pages"])) for s in sources]
    first = next((i for i, h in enumerate(hits, 1) if h), None)
    covered = set().union(*(set(s["pages"]) for s in sources)) if sources else set()
    out = {f"hit@{k}": int(any(hits[:k])) for k in ks}
    out["mrr"] = round(1 / first, 3) if first else 0.0
    out["page_recall"] = round(len(gold & covered) / len(gold), 3)
    if facts:
        context = "\n".join(s["text"] for s in sources)
        out["context_recall"] = round(sum(bool(re.search(f, context, re.I)) for f in facts) / len(facts), 3)
    return out


def auto_label(q: dict, answer: str) -> tuple[str, float]:
    refused = bool(REFUSAL.search(answer))
    if q["type"] == "out_of_scope":
        return ("correct_refusal" if refused else "hallucination"), float(refused)
    facts = q.get("facts") or []
    found = sum(bool(re.search(f, answer, re.I)) for f in facts)
    recall = found / len(facts) if facts else 0.0
    if refused and found == 0:
        return "false_refusal", 0.0
    if recall == 1:
        return "correct", recall
    return ("incomplete" if found else "incorrect"), recall


def write_rows(name: str, rows: list[dict]):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    flat = [{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
             for k, v in r.items() if k != "sources"} for r in rows]
    with open(RESULTS / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)
    print(f"  -> {RESULTS / name}.csv")


# --------------------------------------------------------------------------- retrieval ablation
def run_retrieval(args):
    from rag.chain import RAGPipeline

    base = load_config()
    questions = [q for q in load_questions(ids=args.ids) if q["gold_pages"]]
    grid = list(itertools.product(args.parsers, args.chunkings, args.embeds))
    rows = []
    for parser, chunking, embed in grid:
        cfg = base.with_overrides(pdf_parser=parser, chunking=chunking, embed_model=embed, top_k=5)
        for mode in args.retrieval_modes:
            rag = RAGPipeline(cfg.with_overrides(retrieval_mode=mode))
            print(f"[retrieval] parser={parser} chunking={chunking} embed={embed} mode={mode}")
            for q in questions:
                query = q.get("standalone") or q["question"]
                t0 = time.perf_counter()
                sources = rag.retrieve(query, 5)
                rows.append({
                    "parser": parser, "chunking": chunking, "embed": embed, "retrieval": mode,
                    "id": q["id"], "category": q["category"],
                    **page_metrics(sources, q["gold_pages"], q.get("facts")),
                    "retrieved_pages": [s["pages"] for s in sources],
                    "latency_ms": round((time.perf_counter() - t0) * 1000),
                })
    write_rows("retrieval_ablation", rows)


# --------------------------------------------------------------------------- end-to-end answers
def run_answers(args):
    from rag.chain import RAGPipeline

    cfg = load_config().with_overrides(pdf_parser=args.parser, chunking=args.chunking,
                                       embed_model=args.embed, retrieval_mode=args.retrieval)
    rag = RAGPipeline(cfg)
    questions = load_questions(ids=args.ids)
    for model in args.models:
        name = f"answers__{model.replace(':', '-')}__{cfg.pdf_parser}-{cfg.chunking}-{cfg.retrieval_mode}__{args.mode}"
        print(f"[answers] model={model} mode={args.mode} collection={cfg.collection_name}")
        rag.llm(model).invoke("Reply with OK.")  # warm-up: load the model before timing
        rows = []
        for q in questions:
            result = rag.answer(q["question"], q.get("history"), model=model, mode=args.mode)
            label, recall = auto_label(q, result["answer"])
            rows.append({
                "model": model, "mode": args.mode, "parser": cfg.pdf_parser, "chunking": cfg.chunking,
                "retrieval": cfg.retrieval_mode, "id": q["id"], "category": q["category"], "type": q["type"],
                "question": q["question"], "standalone_query": result["standalone_query"],
                "answer": result["answer"], "gold_answer": q["gold_answer"],
                "auto_label": label, "fact_recall": round(recall, 3),
                **page_metrics(result["sources"], q["gold_pages"], q.get("facts")),
                "retrieved_pages": [s["pages"] for s in result["sources"]],
                "retrieval_ms": result["latency_ms"]["retrieval"],
                "generation_ms": result["latency_ms"]["generation"],
                "sources": result["sources"],
            })
            print(f"  {q['id']:<4} {label:<16} {result['latency_ms']['generation']:>6} ms  {result['answer'][:90]!r}")
        write_rows(name, rows)


# --------------------------------------------------------------------------- report
def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _pct(n, d):
    return f"{100 * n / d:.0f}%" if d else "-"


def run_report(args):
    lines = ["# Evaluation summary", "", "_Generated by `python -m eval.run_eval report`._", ""]

    ret = RESULTS / "retrieval_ablation.jsonl"
    if ret.exists():
        rows = _read(ret)
        groups = defaultdict(list)
        for r in rows:
            groups[(r["parser"], r["chunking"], r["embed"], r["retrieval"])].append(r)
        lines += ["## Retrieval ablation (top-5, questions with gold pages)", "",
                  "| Parser | Chunking | Embedding | Retrieval | Context recall@5 | Hit@1 | Hit@3 | Hit@5 | MRR | Gold-page recall |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        mean_of = lambda g, k: statistics.mean(r[k] for r in g if k in r)
        for (p, c, e, m), g in sorted(groups.items(), key=lambda kv: -mean_of(kv[1], "context_recall")):
            lines.append(f"| {p} | {c} | {e} | {m} | **{mean_of(g, 'context_recall'):.2f}** | {mean_of(g, 'hit@1'):.2f} | "
                         f"{mean_of(g, 'hit@3'):.2f} | {mean_of(g, 'hit@5'):.2f} | {mean_of(g, 'mrr'):.3f} | "
                         f"{mean_of(g, 'page_recall'):.2f} |")
        lines.append("")

    answer_files = sorted(RESULTS.glob("answers__*.jsonl"))
    if answer_files:
        manual_path = HERE / "manual_labels.yaml"
        manual = yaml.safe_load(manual_path.read_text()) if manual_path.exists() else {}
        labels = ["correct", "incomplete", "incorrect", "false_refusal", "correct_refusal", "hallucination"]
        score_of = lambda c, n: (c["correct"] + c["correct_refusal"] + 0.5 * c["incomplete"]) / n
        lines += ["## End-to-end answers", "",
                  "Label counts use the manually reviewed label where a review exists (column *Reviewed*),",
                  "otherwise the automatic label.", "",
                  "| Model | Prompt | Config | " + " | ".join(labels) + " | Score* | Auto score | Reviewed | Gen. median (s) |",
                  "|---|---|---|" + "---|" * len(labels) + "---|---|---|---|"]
        by_cat_lines = []
        for f in answer_files:
            rows = _read(f)
            n = len(rows)
            reviewed = f.stem in manual
            overrides = manual.get(f.stem) or {}
            for r in rows:
                r["label"] = overrides.get(r["id"], {}).get("label", r["auto_label"])
            cnt = Counter(r["label"] for r in rows)
            auto = Counter(r["auto_label"] for r in rows)
            med = statistics.median(r["generation_ms"] for r in rows) / 1000
            r0 = rows[0]
            config = f"{r0['parser']}/{r0['chunking']}/{r0['retrieval']}"
            lines.append(f"| {r0['model']} | {r0['mode']} | {config} | "
                         + " | ".join(str(cnt[l]) for l in labels)
                         + f" | **{score_of(cnt, n):.2f}** | {score_of(auto, n):.2f} | {'yes' if reviewed else 'no'} | {med:.1f} |")
            cats = defaultdict(list)
            for r in rows:
                cats[r["category"]].append(r)
            by_cat_lines.append((f"{r0['model']} · {r0['mode']} · {config}", {
                c: _pct(sum(r["label"] in ("correct", "correct_refusal") for r in g), len(g)) for c, g in cats.items()}))
        lines += ["", "*Score = (correct + correct_refusal + 0.5 x incomplete) / questions.", ""]
        cats = sorted({c for _, d in by_cat_lines for c in d})
        lines += ["## Fully correct answers by category", "",
                  "| Run | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
        for run, d in by_cat_lines:
            lines.append(f"| {run} | " + " | ".join(d.get(c, "-") for c in cats) + " |")
        lines.append("")

    out = RESULTS / "summary.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n-> {out}")


# --------------------------------------------------------------------------- CLI
def main(argv=None):
    csv_list = lambda s: [x.strip() for x in s.split(",") if x.strip()]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("retrieval")
    r.add_argument("--parsers", type=csv_list, default=["pypdf", "pymupdf"])
    r.add_argument("--chunkings", type=csv_list, default=["fixed", "structure"])
    r.add_argument("--embeds", type=csv_list, default=["nomic-embed-text", "mxbai-embed-large"])
    r.add_argument("--retrieval-modes", type=csv_list, default=["dense", "hybrid"])
    r.add_argument("--ids", type=csv_list)
    r.set_defaults(func=run_retrieval)

    a = sub.add_parser("answers")
    a.add_argument("--models", type=csv_list, default=["qwen2.5:3b"])
    a.add_argument("--mode", choices=["strict", "fallback"], default="strict")
    a.add_argument("--parser")
    a.add_argument("--chunking")
    a.add_argument("--embed")
    a.add_argument("--retrieval")
    a.add_argument("--ids", type=csv_list)
    a.set_defaults(func=run_answers)

    sub.add_parser("report").set_defaults(func=run_report)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("rag.index").setLevel(logging.INFO)
    args.func(args)


if __name__ == "__main__":
    main()

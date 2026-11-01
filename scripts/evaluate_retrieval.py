"""Evaluate retrieval quality against the QA set using the app's real pipeline.

Runs the same hybrid (vector + BM25, RRF-fused) retrieval the tutor uses, with
no LLM call, and reports:

- recall@k: share of questions with a relevant chunk in the top k
- MRR: mean reciprocal rank of the first relevant chunk

A chunk is relevant when it comes from the QA item's `source` file (and `page`,
if given). Items without a `source` fall back to answer matching: the chunk
must contain at least --match-threshold of the answer's content words. Adding
`source` to QA items gives far more reliable numbers.

Usage:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --mode vector         # compare against vector-only
    python scripts/evaluate_retrieval.py --limit 20 --min-recall 0.5   # CI gate
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vet_ai_tutor.config import (  # noqa: E402
    EMBEDDING_MODEL, RETRIEVER_K, VECTOR_BACKEND, VECTOR_PATH,
)

_WORD_RE = re.compile(r"\w+")
_STOPWORDS = {
    "the", "and", "are", "for", "from", "that", "this", "with", "which", "into",
    "such", "their", "they", "have", "has", "been", "were", "was", "also", "than",
}


def content_words(text: str) -> set:
    return {w for w in _WORD_RE.findall((text or "").lower()) if len(w) > 3 and w not in _STOPWORDS}


def is_relevant(doc, item: dict, match_threshold: float) -> bool:
    meta = getattr(doc, "metadata", {}) or {}
    if item.get("source"):
        source = str(meta.get("source") or meta.get("source_file") or "")
        if os.path.basename(source) != os.path.basename(item["source"]):
            return False
        page = item.get("page")
        return page is None or str(meta.get("page", "")) == str(page)
    expected = content_words(item.get("answer", ""))
    if not expected:
        return False
    found = content_words(getattr(doc, "page_content", ""))
    return len(expected & found) / len(expected) >= match_threshold


def first_relevant_rank(docs, item: dict, match_threshold: float):
    for rank, doc in enumerate(docs, start=1):
        if is_relevant(doc, item, match_threshold):
            return rank
    return None


def summarize(ranks: list, k: int) -> dict:
    total = len(ranks)
    hits = sum(1 for r in ranks if r is not None and r <= k)
    mrr = sum(1.0 / r for r in ranks if r is not None) / total if total else 0.0
    return {"total": total, "hits": hits, "recall_at_k": hits / total if total else 0.0, "mrr": mrr}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qa", default="data/eval/qa_pairs.json", help="QA JSON file")
    p.add_argument("--k", type=int, default=RETRIEVER_K, help="retrieval k (top-k)")
    p.add_argument("--mode", choices=("hybrid", "vector"), default="hybrid")
    p.add_argument("--match-threshold", type=float, default=0.5,
                   help="answer-word overlap needed when an item has no `source` (0-1)")
    p.add_argument("--limit", type=int, help="evaluate only the first N items (quick CI slice)")
    p.add_argument("--min-recall", type=float, help="exit with status 1 if recall@k is below this")
    args = p.parse_args()

    qa_path = Path(args.qa)
    if not qa_path.exists():
        raise SystemExit(f"QA file not found: {qa_path}")
    qas = json.loads(qa_path.read_text(encoding="utf8"))[: args.limit]
    print(f"Loaded {len(qas)} QA pairs from {qa_path}")

    from vet_ai_tutor.core import Tutor
    tutor = Tutor(llm=None, vector_path=VECTOR_PATH)  # retrieval only; no LLM calls
    if tutor.db is None:
        raise SystemExit("Vectorstore not loaded. Run scripts/rebuild_vectorstore.py first.")
    if tutor.index_warning:
        print(f"WARNING: {tutor.index_warning}")

    print(f"\nEvaluating {args.mode} retrieval, k={args.k}, {tutor.doc_count} chunks "
          f"({EMBEDDING_MODEL} + {VECTOR_BACKEND})")
    print("-" * 70)
    results, ranks = [], []
    t0 = time.time()
    for item in qas:
        q = item.get("question", "")
        docs = tutor.retrieve(q, args.k) if args.mode == "hybrid" else tutor._vector_search(q, args.k)
        rank = first_relevant_rank(docs, item, args.match_threshold)
        ranks.append(rank)
        status = f"HIT@{rank}" if rank else "MISS"
        print(f"[{item.get('id')}] {status:7} {q[:70]}")
        results.append({
            "id": item.get("id"), "question": q, "rank": rank,
            "matched_by": "source" if item.get("source") else "answer_overlap",
            "retrieved": [
                {"source": (getattr(d, "metadata", {}) or {}).get("source", ""),
                 "chunk": (getattr(d, "metadata", {}) or {}).get("chunk")}
                for d in docs
            ],
        })
    elapsed = time.time() - t0

    summary = summarize(ranks, args.k)
    by_source = sum(1 for item in qas if item.get("source"))
    print(f"\n{'=' * 70}")
    print(f"RETRIEVAL SUMMARY ({args.mode})")
    print(f"{'=' * 70}")
    print(f"  Questions:        {summary['total']} ({by_source} matched by source, "
          f"{summary['total'] - by_source} by answer overlap)")
    print(f"  Recall@{args.k}:         {summary['hits']}/{summary['total']} = {summary['recall_at_k']:.1%}")
    print(f"  MRR:              {summary['mrr']:.3f}")
    print(f"  Avg latency:      {elapsed / max(1, len(qas)):.2f}s per query")

    out_dir = Path("data/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_{args.mode}_k{args.k}_{int(time.time())}.json"
    out_path.write_text(json.dumps({
        "config": {"mode": args.mode, "k": args.k, "embedding_model": EMBEDDING_MODEL,
                   "chunks": tutor.doc_count, "match_threshold": args.match_threshold},
        "summary": summary, "results": results,
    }, indent=2), encoding="utf8")
    print(f"\n  Results saved: {out_path}")

    if args.min_recall is not None and summary["recall_at_k"] < args.min_recall:
        print(f"FAIL: recall@{args.k} {summary['recall_at_k']:.1%} is below {args.min_recall:.1%}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

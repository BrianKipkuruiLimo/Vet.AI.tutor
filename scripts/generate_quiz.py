#!/usr/bin/env python3
"""Generate draft quiz questions from passages in the knowledge index.

Picks well-filled passages spread across sources, asks the local LLM for one
multiple-choice question per passage (Ollama JSON mode), and rejects any
that fail validation (4 distinct options, one correct answer supported by
the passage, no "all of the above"). Valid drafts go to data/quiz/pending.json
for review with scripts/review_quiz.py. Progress is saved after every question.

About 20-40 s per question on a laptop CPU; run it in batches.

Usage:
    python scripts/generate_quiz.py --limit 20
    python scripts/generate_quiz.py --sources "babesiosis,mastitis" --per-source 3
    python scripts/generate_quiz.py --model qwen3:4b --limit 10   # better questions, slower
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from vet_ai_tutor.config import LLM_MODEL, VECTOR_PATH, project_path  # noqa: E402
from vet_ai_tutor.quiz import generate_question  # noqa: E402

PENDING = PROJECT_ROOT / "data" / "quiz" / "pending.json"
MIN_PASSAGE_CHARS = 400


def load_pending(path: Path = PENDING) -> list:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def save_pending(items: list, path: Path = PENDING) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def source_of(meta: dict) -> str:
    """Rebuilt chunks store `source`; staged web/document records store `source_file`."""
    return (meta or {}).get("source") or (meta or {}).get("source_file") or "unknown"


def passage_key(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def select_passages(records: list, per_source: int, sources: list, used: set, limit: int,
                    min_chars: int = MIN_PASSAGE_CHARS) -> list:
    """Evenly spaced, well-filled passages from each source, skipping used ones."""
    by_source: dict = {}
    for text, meta in records:
        source = source_of(meta)
        if sources and not any(s in source.lower() or s in (meta.get("source_title", "")).lower()
                               for s in sources):
            continue
        if len(text or "") >= min_chars and passage_key(text) not in used:
            by_source.setdefault(source, []).append((text, meta))
    per_source_picks = []
    for source in sorted(by_source):
        chunks = by_source[source]
        step = max(1, len(chunks) // per_source)
        per_source_picks.append(chunks[::step][:per_source])
    # Round-robin across sources so a limited run still covers many lectures.
    picked = [picks[i] for i in range(per_source) for picks in per_source_picks if i < len(picks)]
    return picked[:limit] if limit else picked


def read_index(index_dir: str) -> list:
    import chromadb

    client = chromadb.PersistentClient(path=project_path(index_dir))
    records = []
    for collection in client.list_collections():
        offset = 0
        while True:
            batch = collection.get(include=["documents", "metadatas"], limit=1000, offset=offset)
            docs = batch.get("documents") or []
            records.extend(zip(docs, batch.get("metadatas") or [{}] * len(docs), strict=True))
            if len(docs) < 1000:
                break
            offset += len(docs)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=20, help="questions to attempt (0 = no limit)")
    parser.add_argument("--per-source", type=int, default=2, help="passages per source document")
    parser.add_argument("--sources", default="", help="comma-separated filename/title filters")
    parser.add_argument("--model", default=LLM_MODEL, help="Ollama model (default: the tutor's model)")
    parser.add_argument("--min-chars", type=int, default=MIN_PASSAGE_CHARS,
                        help="skip passages shorter than this (too little to ask about)")
    args = parser.parse_args()

    from vet_ai_tutor.llm_adapter import build_llm
    llm = build_llm(args.model, num_predict=512, temperature=0.4, format="json")
    pending = load_pending()
    used = {passage_key(q["passage"]) for q in pending}
    filters = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
    passages = select_passages(read_index(VECTOR_PATH), args.per_source, filters, used, args.limit,
                               args.min_chars)
    if not passages:
        raise SystemExit("No unused passages match. Build the index first or relax --sources.")

    made = 0
    t0 = time.time()
    for i, (text, meta) in enumerate(passages, start=1):
        label = source_of(meta)
        try:
            entry, problems = generate_question(llm, text, meta)
        except Exception as exc:  # keep going if one generation fails
            entry, problems = None, [f"generation failed: {exc}"]
        if entry:
            entry["passage"] = text.strip()  # keep the full passage for review and dedup
            pending.append(entry)
            save_pending(pending)
            made += 1
            print(f"[{i}/{len(passages)}] ✓ {label}: {entry['question']}")
        else:
            print(f"[{i}/{len(passages)}] ✗ {label}: {'; '.join(problems)}")
    print(f"\n{made}/{len(passages)} questions passed validation in {time.time() - t0:.0f}s "
          f"-> {PENDING}. Review them with: python scripts/review_quiz.py")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Attach the correct source passage to each evaluation question.

For every QA item without a `source`, this shows the question, the reference
answer and the top passages the current index retrieves. A reviewer who knows
the material picks the passage that actually answers the question; its file
and page/slide are saved as `source` / `page` (or `slide`). Progress is saved
after every answer, so the review can be stopped and resumed.

Keys:  1-9  pick that passage      n  none of them is correct
       m    type a source manually  s  skip for now      q  save and quit

Usage:
    python scripts/review_qa_sources.py
    python scripts/review_qa_sources.py --candidates 5 --all   # re-review everything
    python scripts/review_qa_sources.py --add                  # write new questions
"""
import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_QA = PROJECT_ROOT / "data" / "eval" / "qa_pairs.json"


def load_items(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def save_items(path: Path, items: list) -> None:
    """Write atomically so an interrupted save never corrupts the QA file."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def needs_review(item: dict, review_all: bool) -> bool:
    return review_all or ("source" not in item and not item.get("source_reviewed"))


def location(doc) -> dict:
    meta = getattr(doc, "metadata", {}) or {}
    loc = {"source": meta.get("source") or meta.get("source_file") or "unknown"}
    for key in ("page", "slide"):
        if meta.get(key) not in (None, ""):
            loc[key] = meta[key]
    return loc


def apply_choice(item: dict, choice: str, candidates: list, manual: str = "") -> bool:
    """Update `item` for a reviewer's choice. Returns False if the choice is invalid."""
    choice = choice.strip().lower()
    if choice.isdigit() and 1 <= int(choice) <= len(candidates):
        for key in ("page", "slide"):
            item.pop(key, None)
        item.update(location(candidates[int(choice) - 1]))
        item["source_reviewed"] = True
        return True
    if choice == "n":
        # Retrieval missed it; keep the item but record that no candidate was right.
        item["source_reviewed"] = True
        item["retrieval_missed"] = True
        return True
    if choice == "m" and manual.strip():
        parts = manual.strip().rsplit("#", 1)
        item["source"] = parts[0].strip()
        if len(parts) == 2 and parts[1].strip().isdigit():
            item["page"] = int(parts[1])
        item["source_reviewed"] = True
        return True
    return False


def _print_candidates(item: dict, candidates: list) -> None:
    print("\n" + "=" * 78)
    print(f"[{item.get('id')}] {item.get('question')}")
    print(textwrap.fill(f"Reference answer: {item.get('answer', '')}", 78, subsequent_indent="  "))
    for i, doc in enumerate(candidates, start=1):
        loc = location(doc)
        where = loc["source"] + (f" p.{loc['page']}" if "page" in loc else "") + \
            (f" slide {loc['slide']}" if "slide" in loc else "")
        snippet = " ".join((doc.page_content or "").split())[:300]
        print(f"\n  {i}. {where}\n" + textwrap.indent(textwrap.fill(snippet, 72), "     "))


def review(items: list, retrieve, path: Path, n_candidates: int, review_all: bool) -> int:
    reviewed = 0
    for item in items:
        if not needs_review(item, review_all):
            continue
        candidates = retrieve(item["question"], n_candidates)
        _print_candidates(item, candidates)
        while True:
            choice = input(f"\nPick 1-{len(candidates)}, n(one), m(anual), s(kip), q(uit): ").strip().lower()
            if choice == "q":
                save_items(path, items)
                return reviewed
            if choice == "s":
                break
            manual = input("Source file (optionally file.pdf#page): ") if choice == "m" else ""
            if apply_choice(item, choice, candidates, manual):
                reviewed += 1
                save_items(path, items)
                break
            print("Invalid choice.")
    return reviewed


def add_questions(items: list, path: Path) -> None:
    next_id = 1 + max((int(str(i.get("id", "q0")).lstrip("q") or 0) for i in items
                       if str(i.get("id", "")).lstrip("q").isdigit()), default=0)
    print("Add questions (empty question to stop). Review their sources afterwards.")
    while (question := input("\nQuestion: ").strip()):
        answer = input("Reference answer: ").strip()
        items.append({"id": f"q{next_id}", "question": question, "answer": answer})
        next_id += 1
        save_items(path, items)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qa", default=str(DEFAULT_QA))
    parser.add_argument("--candidates", type=int, default=3, help="passages to show per question (max 9)")
    parser.add_argument("--all", action="store_true", help="re-review items that already have a source")
    parser.add_argument("--add", action="store_true", help="add new questions before reviewing")
    args = parser.parse_args()

    path = Path(args.qa)
    items = load_items(path)
    if args.add:
        add_questions(items, path)

    from vet_ai_tutor.config import VECTOR_PATH
    from vet_ai_tutor.core import Tutor
    tutor = Tutor(llm=None, vector_path=VECTOR_PATH)  # retrieval only
    if tutor.db is None:
        raise SystemExit("Vectorstore not loaded. Build the index first (scripts/rebuild_vectorstore.py).")
    if tutor.index_warning:
        print(f"WARNING: {tutor.index_warning}")

    pending = sum(needs_review(i, args.all) for i in items)
    print(f"{pending} of {len(items)} questions need a source.")
    done = review(items, tutor.retrieve, path, min(max(args.candidates, 1), 9), args.all)
    with_source = sum(1 for i in items if i.get("source"))
    print(f"\nReviewed {done} question(s) this session; {with_source}/{len(items)} now have a source. Saved {path}.")


if __name__ == "__main__":
    main()

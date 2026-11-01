#!/usr/bin/env python3
"""Review draft quiz questions, then publish the approved ones.

Shows each pending question with its source passage and the option the model
marked correct. Keys:
    a  approve          r  reject
    1-4  approve with a different correct option
    s  skip for now     q  save and quit

Progress is saved after every decision. Publishing copies approved questions
into <index>/quiz_bank.json, which ships with knowledge patches.

Usage:
    python scripts/review_quiz.py
    python scripts/review_quiz.py --publish
"""
import argparse
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from vet_ai_tutor.quiz import bank_path, save_bank  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_quiz import PENDING, load_pending, save_pending  # noqa: E402


def apply_decision(entry: dict, key: str) -> bool:
    """Record a reviewer decision on `entry`. Returns False for an invalid key."""
    key = key.strip().lower()
    stamp = datetime.now(timezone.utc).isoformat()
    if key == "a":
        entry.update(status="approved", reviewed_at=stamp)
    elif key == "r":
        entry.update(status="rejected", reviewed_at=stamp)
    elif key in {"1", "2", "3", "4"}:
        entry.update(status="approved", answer_index=int(key) - 1, reviewed_at=stamp)
    else:
        return False
    return True


def show(entry: dict) -> None:
    where = entry["source"] + (f" p.{entry['page']}" if "page" in entry else "") + \
        (f" slide {entry['slide']}" if "slide" in entry else "")
    print("\n" + "=" * 78)
    print(f"Source: {where}")
    print(textwrap.indent(textwrap.fill(" ".join(entry["passage"].split())[:900], 74), "  │ "))
    print(f"\nQ: {entry['question']}")
    for i, option in enumerate(entry["options"]):
        marker = "✓" if i == entry["answer_index"] else " "
        print(f"  {marker} {i + 1}. {option}")
    print(textwrap.fill(f"Explanation: {entry['explanation']}", 78, subsequent_indent="  "))


def review(items: list, path: Path = PENDING) -> int:
    done = 0
    for entry in items:
        if entry.get("status") != "pending_review":
            continue
        show(entry)
        while True:
            key = input("\na(pprove) r(eject) 1-4 (approve with that answer) s(kip) q(uit): ").strip().lower()
            if key == "q":
                return done
            if key == "s":
                break
            if apply_decision(entry, key):
                save_pending(items, path)
                done += 1
                break
            print("Invalid key.")
    return done


def publish(items: list, path: str) -> int:
    """Merge approved questions into the bank (existing entries kept, deduplicated by id)."""
    import json
    import os

    existing = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            existing = json.load(f).get("questions", [])
    by_id = {q["id"]: q for q in existing}
    for entry in items:
        if entry.get("status") == "approved":
            by_id[entry["id"]] = entry
        elif entry.get("status") == "rejected":
            by_id.pop(entry["id"], None)
    save_bank(sorted(by_id.values(), key=lambda q: (q["source"], q["id"])), path)
    return sum(1 for q in by_id.values() if q.get("status") == "approved")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--publish", action="store_true", help="copy approved questions into the index's bank")
    args = parser.parse_args()
    items = load_pending()
    if args.publish:
        count = publish(items, bank_path())
        print(f"Quiz bank now has {count} approved questions: {bank_path()}")
        print("Restart the app (or apply a knowledge patch) to serve them.")
        return
    pending = sum(1 for q in items if q.get("status") == "pending_review")
    print(f"{pending} question(s) awaiting review.")
    done = review(items)
    print(f"\nReviewed {done} question(s). Publish approved ones with: python scripts/review_quiz.py --publish")


if __name__ == "__main__":
    main()

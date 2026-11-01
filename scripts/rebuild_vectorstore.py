#!/usr/bin/env python3
"""Rebuild the knowledge index from data/raw_notes/.

Only files with a reviewed entry in data/source_metadata.json are indexed.
The new index is built beside the current one and swapped in atomically,
with a backup. The full report is written to archive/rebuild_summary.json.

Usage:
    python scripts/rebuild_vectorstore.py --dry-run   # report only, no embedding
    python scripts/rebuild_vectorstore.py

Prerequisites:
    1. Ollama running:   ollama serve
    2. Embedding model:  ollama pull nomic-embed-text
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from vet_ai_tutor.config import RAW_DIR, SOURCE_CATALOG_PATH  # noqa: E402
from vet_ai_tutor.ingestion.provenance import ProvenanceError  # noqa: E402
from vet_ai_tutor.ingestion.rebuild import SUMMARY_PATH, rebuild  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Load and validate only; do not embed")
    parser.add_argument("--raw-dir", default=RAW_DIR)
    parser.add_argument("--catalog", default=SOURCE_CATALOG_PATH)
    parser.add_argument("--quiet", action="store_true", help="Only print the summary")
    args = parser.parse_args()
    try:
        report = rebuild(dry_run=args.dry_run, raw_dir=args.raw_dir, catalog_path=args.catalog,
                         progress=(lambda _msg: None) if args.quiet else print)
    except (ProvenanceError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Rebuild failed: {exc}\n")
    print("\n" + "\n".join(report.summary_lines()))
    if report.dry_run:
        print("\nDry run: the index was not changed.")
    else:
        print(f"\nNew index installed. Previous index backed up to: {report.backup_path or '(none)'}")
    print(f"Full report: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

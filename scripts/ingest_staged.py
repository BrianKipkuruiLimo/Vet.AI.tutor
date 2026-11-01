#!/usr/bin/env python3
"""Ingest Staged Data — Move approved data from staging into the vector database.

Reads JSON files from data/staging/ that have been reviewed and approved,
creates semantically chunked documents with rich metadata (species, topic,
source tier, provenance), and adds them to the active vector database.

Usage:
    # Ingest all approved staging files:
    python scripts/ingest_staged.py

    # Ingest only specific files:
    python scripts/ingest_staged.py --file data/staging/giraffe_abc123_20261001.json

    # Approve one reviewed file:
    python scripts/ingest_staged.py --approve giraffe_abc123_20261001.json

    # Review staging files (dry run):
    python scripts/ingest_staged.py --review

Prerequisites:
    1. Ollama running:   ollama serve
    2. Embedding model:  ollama pull nomic-embed-text
    3. Staging files from harvester.py or manual placement
"""

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vet_ai_tutor.config import VECTOR_PATH  # noqa: E402
from vet_ai_tutor.embeddings import load_embeddings  # noqa: E402,F401  (used by dev_dashboard)
from vet_ai_tutor.ingestion.indexer import open_vectorstore  # noqa: E402
from vet_ai_tutor.ingestion.staging import (  # noqa: E402,F401  (re-exported for dev_dashboard/tests)
    STAGING_DIR, TOPICS, approve_file, create_semantic_documents, ingest_file, list_staging_files,
)


def get_vectorstore(embeddings):
    """Open or create the active vector database."""
    os.makedirs(VECTOR_PATH, exist_ok=True)
    return open_vectorstore(embeddings)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest approved staging files into the vector database.",
    )
    parser.add_argument("--file", type=str, help="Ingest a specific staging file.")
    parser.add_argument("--approve", type=str, help="Approve one reviewed staging JSON by filename.")
    parser.add_argument("--show", type=str, help="Print one staging JSON for full review.")
    parser.add_argument("--review", action="store_true", help="Review staging files without ingesting (dry run).")
    args = parser.parse_args()

    if args.approve:
        candidate = os.path.abspath(os.path.join(STAGING_DIR, os.path.basename(args.approve)))
        if os.path.commonpath([candidate, os.path.abspath(STAGING_DIR)]) != os.path.abspath(STAGING_DIR):
            raise ValueError("Approval path must be inside the staging directory")
        approve_file(candidate)
        logger.info("Approved %s. Run the command again to ingest it.", candidate)
        return

    if args.show:
        candidate = os.path.abspath(os.path.join(STAGING_DIR, os.path.basename(args.show)))
        if os.path.commonpath([candidate, os.path.abspath(STAGING_DIR)]) != os.path.abspath(STAGING_DIR):
            raise ValueError("Review path must be inside the staging directory")
        with open(candidate, encoding="utf-8") as f:
            print(json.dumps(json.load(f), indent=2, ensure_ascii=False))
        return

    # Review mode
    if args.review:
        files = list_staging_files()
        if not files:
            print("No files in staging area.")
            return

        print(f"\n{'='*80}")
        print(f"{'Status':<18} {'Species':<15} {'Tier':<10} {'Year':<6} {'Title / filename'}")
        print(f"{'='*80}")
        for f in files:
            print(f"{f['status']:<18} {f['species']:<15} {f['source_tier']:<10} {f['publication_year']:<6} {f['source_title'] or f['filename']}")
        print(f"{'='*80}")
        print(f"Total: {len(files)} files")
        print("Inspect a record: python scripts/ingest_staged.py --show <filename.json>")
        print("Review provenance and extracted content before approval.")
        print("Approve one file: python scripts/ingest_staged.py --approve <filename.json>")
        return

    # Load embeddings
    embeddings = load_embeddings()
    if embeddings is None:
        logger.error("Cannot load embeddings. Aborting.")
        sys.exit(1)

    # Open vectorstore
    db = get_vectorstore(embeddings)

    # Determine which files to ingest
    if args.file:
        files_to_ingest = [args.file]
    else:
        staged = list_staging_files(status_filter="approved")
        files_to_ingest = [f["path"] for f in staged]

    if not files_to_ingest:
        print("No approved files to ingest.")
        print("Run with --review to see staging files, then approve individual reviewed records.")
        return

    # Ingest each file
    total_docs = 0
    for filepath in files_to_ingest:
        try:
            db, count = ingest_file(filepath, embeddings, db)
            total_docs += count
        except Exception as exc:
            logger.error("Rejected %s: %s", filepath, exc)

    print(f"\n{'='*50}")
    print(f"Ingestion complete: {total_docs} documents added to vectorstore.")
    print(f"Vector database: {os.path.abspath(VECTOR_PATH)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

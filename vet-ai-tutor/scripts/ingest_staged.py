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

    # Approve all pending files and ingest:
    python scripts/ingest_staged.py --approve-all

    # Review staging files (dry run):
    python scripts/ingest_staged.py --review

Prerequisites:
    1. Ollama running:   ollama serve
    2. Embedding model:  ollama pull nomic-embed-text
    3. Staging files from harvester.py or manual placement
"""

import os
import sys
import json
import glob
import shutil
import argparse
import logging
import hashlib
import re
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vet_ai_tutor.config import (
    VECTOR_PATH, EMBEDDING_MODEL, EMBEDDING_BACKEND,
    CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_BACKEND,
    WHITELIST_PATH, OFFLINE_ONLY,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
STAGING_DIR = os.path.join(PROJECT_ROOT, "data", "staging")
INGESTED_DIR = os.path.join(PROJECT_ROOT, "data", "staging", "ingested")
INGEST_LOG = os.path.join(PROJECT_ROOT, "data", "staging", "ingest_log.json")
INGESTED_HASHES = os.path.join(PROJECT_ROOT, "data", "staging", "ingested_hashes.json")

# The topics we extract from structured data
TOPICS = [
    "taxonomy", "biology", "development", "diseases",
    "nutrition", "behavior", "conservation", "veterinary_notes",
]


# ---------------------------------------------------------------------------
# Create semantic documents from structured data
# ---------------------------------------------------------------------------
def create_semantic_documents(staging_doc: dict, source_filename: str):
    """Convert a staged JSON document into semantically chunked LangChain Documents.

    Instead of blindly chunking all text, we create one document per topic
    (taxonomy, diseases, biology, etc.) with rich metadata.
    """
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    extracted = staging_doc.get("extracted_data", {})
    species = staging_doc.get("species", "unknown")
    category = staging_doc.get("category", "unknown")
    source_url = staging_doc.get("source_url", "")
    source_tier = staging_doc.get("source_tier", "unverified")
    source_domain = staging_doc.get("source_domain", "")
    date_harvested = staging_doc.get("date_harvested", "")
    provenance = {
        "source_title": staging_doc.get("source_title", ""),
        "author": staging_doc.get("author", ""),
        "journal": staging_doc.get("journal", ""),
        "publication_year": str(staging_doc.get("publication_year", "")),
    }
    source_hash = hashlib.sha256(json.dumps(
        {"url": source_url, "data": extracted}, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")).hexdigest()

    documents = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    # Base metadata for all chunks from this source
    base_meta = {
        "species": species,
        "category": category,
        "source_url": source_url,
        "source_tier": source_tier,
        "source_domain": source_domain,
        "source_file": source_filename,
        **provenance,
        "source_hash": source_hash,
        "date_harvested": date_harvested,
        "date_ingested": datetime.now(timezone.utc).isoformat(),
    }

    # Add common name and scientific name if available
    common_name = extracted.get("common_name", species)
    scientific_name = extracted.get("scientific_name")

    # Create a document for each topic that has content
    for topic in TOPICS:
        content = extracted.get(topic)
        if not content or content == "null":
            continue

        # Prefix the content with species context for better retrieval
        prefix = f"{common_name}"
        if scientific_name:
            prefix += f" ({scientific_name})"
        full_text = f"{prefix} — {topic.replace('_', ' ').title()}:\n{content}"

        # Split long topics into sub-chunks
        topic_chunks = splitter.split_text(full_text)
        for i, chunk_text in enumerate(topic_chunks):
            meta = {
                **base_meta,
                "topic": topic,
                "common_name": common_name,
                "scientific_name": scientific_name or "",
                "chunk_index": i,
                "total_chunks": len(topic_chunks),
            }
            documents.append(Document(page_content=chunk_text, metadata=meta))

    # Also create a summary document for general questions
    summary = extracted.get("summary")
    if summary:
        prefix = f"{common_name}"
        if scientific_name:
            prefix += f" ({scientific_name})"
        summary_text = f"{prefix} — Overview:\n{summary}"
        meta = {**base_meta, "topic": "summary", "common_name": common_name,
                "scientific_name": scientific_name or "", "chunk_index": 0, "total_chunks": 1}
        documents.append(Document(page_content=summary_text, metadata=meta))

    # Handle raw text fallback (when LLM extraction failed)
    raw_text = extracted.get("raw_text")
    if raw_text and extracted.get("_extraction_failed"):
        raw_chunks = splitter.split_text(raw_text)
        for i, chunk_text in enumerate(raw_chunks):
            meta = {
                **base_meta,
                "topic": "raw_unstructured",
                "common_name": common_name,
                "scientific_name": "",
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
            }
            documents.append(Document(page_content=chunk_text, metadata=meta))

    logger.info(
        "Created %d semantic documents from %s (%s)",
        len(documents), source_filename, species,
    )
    return documents


# ---------------------------------------------------------------------------
# Load embeddings
# ---------------------------------------------------------------------------
def load_embeddings():
    """Load the embedding function (same logic as core.py)."""
    if EMBEDDING_BACKEND == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
            emb = OllamaEmbeddings(model=EMBEDDING_MODEL)
            emb.embed_query("test")
            logger.info("Using OllamaEmbeddings: %s", EMBEDDING_MODEL)
            return emb
        except Exception as e:
            logger.warning("OllamaEmbeddings failed: %s", e)

    if OFFLINE_ONLY and EMBEDDING_BACKEND != "huggingface":
        logger.error("Ollama embeddings unavailable and offline mode is enabled; refusing network fallback.")
        return None

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        hf_model = EMBEDDING_MODEL if EMBEDDING_BACKEND == "huggingface" else "sentence-transformers/all-MiniLM-L6-v2"
        emb = HuggingFaceEmbeddings(model_name=hf_model, model_kwargs={"local_files_only": OFFLINE_ONLY})
        logger.info("Using HuggingFaceEmbeddings: %s", hf_model)
        return emb
    except Exception as e:
        logger.error("No embedding backend available: %s", e)
        return None


# ---------------------------------------------------------------------------
# Vectorstore operations
# ---------------------------------------------------------------------------
def get_vectorstore(embeddings):
    """Open or create the vector database."""
    if VECTOR_BACKEND == "chroma":
        try:
            from langchain_chroma import Chroma
            db = Chroma(
                persist_directory=VECTOR_PATH,
                embedding_function=embeddings,
            )
            logger.info("Opened ChromaDB at %s (%d existing docs)", VECTOR_PATH, db._collection.count())
            return db
        except Exception as e:
            logger.error("Failed to open ChromaDB: %s", e)
            return None
    else:
        try:
            from langchain_community.vectorstores import FAISS
            if os.path.exists(VECTOR_PATH):
                db = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)
                logger.info("Loaded existing FAISS index.")
            else:
                # Will be created when first documents are added
                db = None
            return db
        except Exception as e:
            logger.error("Failed to load FAISS: %s", e)
            return None


def add_to_vectorstore(db, documents, embeddings):
    """Add documents to the vectorstore. Handles both Chroma and FAISS."""
    if not documents:
        return db

    BATCH = 100

    if VECTOR_BACKEND == "chroma":
        for i in range(0, len(documents), BATCH):
            batch = documents[i:i + BATCH]
            pct = min(i + BATCH, len(documents))
            logger.info("  Embedding batch %d–%d / %d...", i, pct, len(documents))
            db.add_documents(batch)
        logger.info("Added %d documents to ChromaDB (total: %d)", len(documents), db._collection.count())
        return db
    else:
        from langchain_community.vectorstores import FAISS
        if db is None:
            db = FAISS.from_documents(documents, embeddings)
        else:
            new_db = FAISS.from_documents(documents, embeddings)
            db.merge_from(new_db)
        os.makedirs(VECTOR_PATH, exist_ok=True)
        db.save_local(VECTOR_PATH)
        logger.info("Added %d documents to FAISS.", len(documents))
        return db


# ---------------------------------------------------------------------------
# Review & approve
# ---------------------------------------------------------------------------
def list_staging_files(status_filter: str = None) -> list[dict]:
    """List all staging files with their metadata."""
    if not os.path.exists(STAGING_DIR):
        return []

    files = []
    for fp in sorted(glob.glob(os.path.join(STAGING_DIR, "*.json"))):
        basename = os.path.basename(fp)
        if basename in ("harvest_log.json", "ingest_log.json"):
            continue
        try:
            with open(fp, "r", encoding="utf-8") as f:
                doc = json.load(f)
            status = doc.get("status", "unknown")
            if status_filter and status != status_filter:
                continue
            files.append({
                "path": fp,
                "filename": basename,
                "species": doc.get("species", "?"),
                "category": doc.get("category", "?"),
                "source_url": doc.get("source_url", "?"),
                "source_title": doc.get("source_title", "?"),
                "author": doc.get("author", "?"),
                "journal": doc.get("journal", "?"),
                "publication_year": doc.get("publication_year", "?"),
                "source_tier": doc.get("source_tier", "?"),
                "status": status,
                "date": doc.get("date_harvested", "?"),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return files


def approve_file(filepath: str):
    """Mark a staging file as approved for ingestion."""
    with open(filepath, "r", encoding="utf-8") as f:
        doc = json.load(f)
    required = ("source_title", "author", "journal", "publication_year", "source_url")
    missing = [key for key in required if not str(doc.get(key, "")).strip()]
    year = str(doc.get("publication_year", ""))
    if missing or not re.fullmatch(r"(?:18|19|20|21)\d{2}", year):
        raise ValueError(f"Incomplete provenance: {', '.join(missing) if missing else 'invalid publication year'}")
    if doc.get("source_tier") not in {"tier_1", "tier_2"}:
        raise ValueError("Only reviewed tier_1 and tier_2 sources can be approved")
    if str(doc["source_url"]).startswith("https://"):
        from scripts.harvester import load_whitelist, check_domain
        whitelist = load_whitelist(WHITELIST_PATH)
        if check_domain(doc["source_url"], whitelist) != doc["source_tier"]:
            raise ValueError("HTTPS source is not whitelisted at the declared tier")
    elif not str(doc["source_url"]).startswith("file://"):
        raise ValueError("Source URL must use HTTPS or file://")
    doc["status"] = "approved"
    doc["approved_at"] = datetime.now(timezone.utc).isoformat()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)


def move_to_ingested(filepath: str):
    """Move an ingested file to the ingested/ subdirectory for record-keeping."""
    os.makedirs(INGESTED_DIR, exist_ok=True)
    dest = os.path.join(INGESTED_DIR, os.path.basename(filepath))
    shutil.move(filepath, dest)
    logger.info("Moved to ingested: %s", dest)


def update_ingest_log(filename: str, species: str, doc_count: int, status: str):
    """Append to the ingest log."""
    entry = {
        "file": filename,
        "species": species,
        "documents_created": doc_count,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    log = []
    if os.path.exists(INGEST_LOG):
        with open(INGEST_LOG, "r", encoding="utf-8") as f:
            try:
                log = json.load(f)
            except json.JSONDecodeError:
                log = []
    log.append(entry)
    with open(INGEST_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------
def ingest_file(filepath: str, embeddings, db) -> tuple:
    """Ingest a single staging file into the vectorstore.

    Returns (db, doc_count) — db may be newly created for FAISS.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        staging_doc = json.load(f)

    filename = os.path.basename(filepath)
    status = staging_doc.get("status", "unknown")

    if status != "approved":
        logger.info("Skipping %s (status: %s)", filename, status)
        return db, 0

    required = ("source_title", "author", "journal", "publication_year", "source_url")
    missing = [key for key in required if not str(staging_doc.get(key, "")).strip()]
    year = str(staging_doc.get("publication_year", ""))
    if missing or not re.fullmatch(r"(?:18|19|20|21)\d{2}", year):
        raise ValueError(f"{filename}: incomplete provenance (missing {missing or ['valid publication_year']})")
    if staging_doc.get("source_tier") not in {"tier_1", "tier_2"}:
        raise ValueError(f"{filename}: only reviewed tier_1 and tier_2 sources can be ingested")
    source_url = str(staging_doc["source_url"])
    if not (source_url.startswith("https://") or source_url.startswith("file://")):
        raise ValueError(f"{filename}: source_url must use HTTPS or file://")
    if source_url.startswith("https://"):
        from scripts.harvester import load_whitelist, check_domain
        whitelist = load_whitelist(WHITELIST_PATH)
        if check_domain(source_url, whitelist) != staging_doc["source_tier"]:
            raise ValueError(f"{filename}: source URL is not whitelisted at the declared tier")

    digest = hashlib.sha256(json.dumps(
        {"url": staging_doc["source_url"], "data": staging_doc.get("extracted_data", {})},
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    try:
        with open(INGESTED_HASHES, "r", encoding="utf-8") as f:
            known_hashes = set(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        known_hashes = set()
    if digest in known_hashes:
        raise ValueError(f"{filename}: duplicate content has already been ingested")

    # Create semantic documents
    documents = create_semantic_documents(staging_doc, filename)
    if not documents:
        logger.warning("No documents created from %s", filename)
        update_ingest_log(filename, staging_doc.get("species", "?"), 0, "empty")
        return db, 0

    # Add to vectorstore
    db = add_to_vectorstore(db, documents, embeddings)

    # Move file to ingested and log
    move_to_ingested(filepath)
    update_ingest_log(filename, staging_doc.get("species", "?"), len(documents), "ingested")
    known_hashes.add(digest)
    os.makedirs(os.path.dirname(INGESTED_HASHES), exist_ok=True)
    with open(INGESTED_HASHES, "w", encoding="utf-8") as f:
        json.dump(sorted(known_hashes), f, indent=2)

    return db, len(documents)


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
        with open(candidate, "r", encoding="utf-8") as f:
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
    os.makedirs(VECTOR_PATH, exist_ok=True)
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

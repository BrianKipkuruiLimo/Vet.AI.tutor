"""Staged JSON records: review, approval and incremental ingestion.

Records come from scripts/harvester.py (web pages) and
scripts/ingest_documents.py (local documents). Each must be approved by a
person, carry complete whitelisted provenance and have no unreviewed
prompt-injection warnings before it is added to the active index.
"""
import glob
import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from typing import Optional

from vet_ai_tutor.config import CHUNK_OVERLAP, CHUNK_SIZE, STAGING_DIR as _STAGING_DIR, project_path
from vet_ai_tutor.ingestion.indexer import add_documents
from vet_ai_tutor.ingestion.provenance import load_whitelist, validate_provenance

logger = logging.getLogger(__name__)

STAGING_DIR = project_path(_STAGING_DIR)
INGESTED_DIR = os.path.join(STAGING_DIR, "ingested")
INGEST_LOG = os.path.join(STAGING_DIR, "ingest_log.json")
INGESTED_HASHES = os.path.join(STAGING_DIR, "ingested_hashes.json")
_LOG_FILES = ("harvest_log.json", "ingest_log.json", "ingested_hashes.json")

# The topics we extract from structured data
TOPICS = [
    "taxonomy", "biology", "development", "diseases",
    "nutrition", "behavior", "conservation", "veterinary_notes",
]


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


def validate_record(doc: dict, filename: str) -> dict:
    """Checks a staged record must pass before ingestion."""
    stored = validate_provenance(doc, filename, load_whitelist())
    if doc.get("injection_warnings") and not doc.get("injection_warnings_reviewed"):
        raise ValueError(
            f"{filename}: possible prompt-injection text was flagged. Remove it from the record, "
            "or set \"injection_warnings_reviewed\": true after checking it is harmless."
        )
    return stored


def list_staging_files(status_filter: Optional[str] = None, staging_dir: Optional[str] = None) -> list[dict]:
    """List all staging files with their metadata."""
    staging_dir = staging_dir or STAGING_DIR
    if not os.path.exists(staging_dir):
        return []

    files = []
    for fp in sorted(glob.glob(os.path.join(staging_dir, "*.json"))):
        basename = os.path.basename(fp)
        if basename in _LOG_FILES:
            continue
        try:
            with open(fp, encoding="utf-8") as f:
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
    """Mark a staging file as approved after checking its provenance."""
    with open(filepath, encoding="utf-8") as f:
        doc = json.load(f)
    validate_provenance(doc, os.path.basename(filepath), load_whitelist())
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
        with open(INGEST_LOG, encoding="utf-8") as f:
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
    with open(filepath, encoding="utf-8") as f:
        staging_doc = json.load(f)

    filename = os.path.basename(filepath)
    status = staging_doc.get("status", "unknown")

    if status != "approved":
        logger.info("Skipping %s (status: %s)", filename, status)
        return db, 0

    validate_record(staging_doc, filename)

    digest = hashlib.sha256(json.dumps(
        {"url": staging_doc["source_url"], "data": staging_doc.get("extracted_data", {})},
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    try:
        with open(INGESTED_HASHES, encoding="utf-8") as f:
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
    db = add_documents(db, documents, embeddings)

    # Move file to ingested and log
    move_to_ingested(filepath)
    update_ingest_log(filename, staging_doc.get("species", "?"), len(documents), "ingested")
    known_hashes.add(digest)
    os.makedirs(os.path.dirname(INGESTED_HASHES), exist_ok=True)
    with open(INGESTED_HASHES, "w", encoding="utf-8") as f:
        json.dump(sorted(known_hashes), f, indent=2)

    return db, len(documents)



#!/usr/bin/env python3
"""Ingest Documents — Process PDFs, Word docs, and books into structured staging files.

Reads local document files (PDF, DOCX, PPTX, TXT), extracts text,
optionally uses an LLM to structure the content by topic, and saves
the result to the staging area for review before ingestion.

For books: attempts to detect chapter structure and creates separate
staging files per chapter for finer-grained retrieval.

Usage:
    # Ingest a single PDF:
    python scripts/ingest_documents.py --file "path/to/document.pdf" --species giraffe

    # Ingest an entire folder of documents:
    python scripts/ingest_documents.py --folder "path/to/docs/" --species cattle --category domestic

    # Ingest without LLM structuring (raw text only — faster):
    python scripts/ingest_documents.py --file "doc.pdf" --species giraffe --raw

    # Ingest a book with chapter detection:
    python scripts/ingest_documents.py --file "textbook.pdf" --species multi --book

Prerequisites:
    1. Ollama running (unless --raw):  ollama serve
    2. LLM model (unless --raw):       ollama pull llama3.2:3b
"""

import os
import sys
import re
import json
import hashlib
import argparse
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vet_ai_tutor.config import SKIP_EXTENSIONS, SUPPORTED_EXTENSIONS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
STAGING_DIR = os.path.join(PROJECT_ROOT, "data", "staging")


# ---------------------------------------------------------------------------
# Document loaders
# ---------------------------------------------------------------------------
def load_document(filepath: str) -> list[dict]:
    """Load a document file and return a list of {text, metadata} dicts.

    Each dict represents a logical section (page, slide, etc.).
    """
    ext = os.path.splitext(filepath.lower())[1]

    if ext in SKIP_EXTENSIONS:
        logger.warning("Skipping unsupported file type: %s", ext)
        return []

    if ext not in SUPPORTED_EXTENSIONS:
        logger.warning("Unsupported extension %s for %s", ext, filepath)
        return []

    kind = SUPPORTED_EXTENSIONS[ext]
    filename = os.path.basename(filepath)

    try:
        if kind == "pdf":
            return _load_pdf(filepath, filename)
        elif kind == "docx":
            return _load_docx(filepath, filename)
        elif kind in ("pptx", "ppt"):
            return _load_pptx(filepath, filename)
        elif kind == "txt":
            return _load_txt(filepath, filename)
        elif kind == "doc":
            return _load_docx(filepath, filename)
        elif kind in ("csv", "xlsx"):
            return _load_spreadsheet(filepath, filename)
    except Exception as e:
        logger.error("Failed to load %s: %s", filepath, e)
        return []

    return []


def _load_pdf(filepath: str, filename: str) -> list[dict]:
    """Load PDF and return pages as sections."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError:
        logger.error("pypdf not installed. Install: pip install pypdf")
        return []

    loader = PyPDFLoader(filepath)
    pages = loader.load()
    sections = []
    for i, page in enumerate(pages):
        text = (page.page_content or "").strip()
        if len(text) < 20:
            continue
        sections.append({
            "text": text,
            "metadata": {"page": i + 1, "source": filename},
        })
    logger.info("Loaded %s: %d pages with content", filename, len(sections))
    return sections


def _load_docx(filepath: str, filename: str) -> list[dict]:
    """Load Word document."""
    try:
        from langchain_community.document_loaders import Docx2txtLoader
    except ImportError:
        logger.error("docx2txt not installed. Install: pip install docx2txt")
        return []

    loader = Docx2txtLoader(filepath)
    docs = loader.load()
    sections = []
    for doc in docs:
        text = (doc.page_content or "").strip()
        if len(text) < 20:
            continue
        sections.append({
            "text": text,
            "metadata": {"source": filename},
        })
    return sections


def _load_pptx(filepath: str, filename: str) -> list[dict]:
    """Load PowerPoint file."""
    try:
        from pptx import Presentation
    except ImportError:
        logger.error("python-pptx not installed. Install: pip install python-pptx")
        return []

    prs = Presentation(filepath)
    sections = []
    for slide_num, slide in enumerate(prs.slides, 1):
        parts = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                parts.append(shape.text.strip())
        if parts:
            sections.append({
                "text": "\n".join(parts),
                "metadata": {"slide": slide_num, "source": filename},
            })
    return sections


def _load_txt(filepath: str, filename: str) -> list[dict]:
    """Load plain text file."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read().strip()
    if len(text) < 20:
        return []
    return [{"text": text, "metadata": {"source": filename}}]


def _load_spreadsheet(filepath: str, filename: str) -> list[dict]:
    """Load CSV or Excel file and return rows as sections."""
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not installed. Install: pip install pandas openpyxl")
        return []

    try:
        if filepath.lower().endswith(".csv"):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
    except Exception as e:
        logger.error("Failed to read spreadsheet %s: %s", filename, e)
        return []

    sections = []
    # Fill NA to prevent empty strings or nan showing up
    df = df.fillna("")
    
    # Process each row into a text summary
    for i, row in df.iterrows():
        row_dict = row.to_dict()
        # Convert dictionary to a readable text format: Column: Value
        row_text = "\\n".join([f"{str(k)}: {str(v)}" for k, v in row_dict.items() if str(v).strip()])
        if len(row_text) < 10:
            continue
        sections.append({
            "text": row_text,
            "metadata": {"row": i + 1, "source": filename},
        })
    logger.info("Loaded %s: %d rows with content", filename, len(sections))
    return sections


# ---------------------------------------------------------------------------
# Book chapter detection
# ---------------------------------------------------------------------------
_CHAPTER_PATTERNS = [
    re.compile(r"^(chapter\s+\d+[:\.\s].*)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^(CHAPTER\s+[IVXLCDM]+[:\.\s].*)", re.MULTILINE),
    re.compile(r"^(Part\s+\d+[:\.\s].*)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^(Section\s+\d+[:\.\s].*)", re.IGNORECASE | re.MULTILINE),
]


def detect_chapters(full_text: str) -> list[dict]:
    """Attempt to split a book into chapters based on heading patterns.

    Returns a list of {"title": ..., "text": ...} dicts.
    """
    # Try each pattern
    for pattern in _CHAPTER_PATTERNS:
        matches = list(pattern.finditer(full_text))
        if len(matches) >= 2:
            chapters = []
            for i, match in enumerate(matches):
                title = match.group(1).strip()
                start = match.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
                chapter_text = full_text[start:end].strip()
                if len(chapter_text) > 50:
                    chapters.append({"title": title, "text": chapter_text})
            if chapters:
                logger.info("Detected %d chapters using pattern: %s", len(chapters), pattern.pattern[:40])
                return chapters

    # Fallback: split into ~5000 char sections
    logger.info("No chapter pattern detected; splitting into sections by size.")
    section_size = 5000
    sections = []
    for i in range(0, len(full_text), section_size):
        chunk = full_text[i:i + section_size].strip()
        if len(chunk) > 50:
            sections.append({"title": f"Section {i // section_size + 1}", "text": chunk})
    return sections


# ---------------------------------------------------------------------------
# LLM structuring
# ---------------------------------------------------------------------------
STRUCTURE_PROMPT = """You are a veterinary data extraction assistant. Read the following text
from a document. Extract veterinary information into structured JSON format.

RULES:
- Only extract information explicitly stated in the text.
- If a section has no relevant info, set its value to null.
- Keep each field concise (max 300 words).
- Return ONLY valid JSON, no markdown fences.

OUTPUT FORMAT:
{{
    "common_name": "string or null",
    "scientific_name": "string or null",
    "taxonomy": "string or null",
    "biology": "string or null",
    "development": "string or null",
    "diseases": "string or null",
    "nutrition": "string or null",
    "behavior": "string or null",
    "conservation": "string or null",
    "veterinary_notes": "string or null",
    "summary": "A brief 2-3 sentence summary"
}}

TEXT:
{text}

JSON OUTPUT:"""


def structure_with_llm(text: str, llm_model: str = "llama3.2:3b") -> dict | None:
    """Use Ollama LLM to extract structured data from document text."""
    try:
        from langchain_ollama import OllamaLLM
    except ImportError:
        return None

    # Truncate very long texts
    if len(text) > 10000:
        text = text[:10000] + "\n[... truncated ...]"

    prompt = STRUCTURE_PROMPT.format(text=text)
    try:
        llm = OllamaLLM(model=llm_model, num_predict=2048, timeout=120, num_ctx=8192)
        response = llm.invoke(prompt)
    except Exception as e:
        logger.error("LLM structuring failed: %s", e)
        return None

    try:
        text_resp = response.strip()
        if text_resp.startswith("```"):
            text_resp = text_resp.split("\n", 1)[1]
            if text_resp.endswith("```"):
                text_resp = text_resp[:-3]
        return json.loads(text_resp)
    except json.JSONDecodeError:
        logger.warning("LLM returned invalid JSON.")
        return None


# ---------------------------------------------------------------------------
# Save to staging
# ---------------------------------------------------------------------------
def save_to_staging(
    structured_data: dict,
    species: str,
    category: str,
    source_file: str,
    source_tier: str = "tier_1",
    chapter_title: str = "",
    source_title: str = "",
    author: str = "",
    journal: str = "",
    publication_year: str = "",
) -> str:
    """Save structured document data to the staging area."""
    os.makedirs(STAGING_DIR, exist_ok=True)

    file_hash = hashlib.md5(source_file.encode()).hexdigest()[:8]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_chapter = re.sub(r"[^\w]", "_", chapter_title)[:30] if chapter_title else ""
    suffix = f"_{safe_chapter}" if safe_chapter else ""
    filename = f"{species}_{file_hash}{suffix}_{timestamp}.json"
    filepath = os.path.join(STAGING_DIR, filename)

    staging_doc = {
        "status": "pending_review",
        "species": species.lower(),
        "category": category.lower(),
        "source_url": f"file://{os.path.abspath(source_file)}",
        "source_tier": source_tier,
        "source_domain": "local_document",
        "source_file_original": os.path.basename(source_file),
        "source_title": source_title or os.path.basename(source_file),
        "author": author,
        "journal": journal,
        "publication_year": str(publication_year),
        "chapter_title": chapter_title,
        "date_harvested": datetime.now(timezone.utc).isoformat(),
        "extracted_data": structured_data,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(staging_doc, f, indent=2, ensure_ascii=False)

    logger.info("Saved to staging: %s", filepath)
    return filepath


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def process_document(
    filepath: str,
    species: str,
    category: str = "unknown",
    source_tier: str = "tier_1",
    use_llm: bool = True,
    is_book: bool = False,
    llm_model: str = "llama3.2:3b",
    source_title: str = "",
    author: str = "",
    journal: str = "",
    publication_year: str = "",
):
    """Process a single document file through the ingestion pipeline."""
    logger.info("Processing: %s", filepath)

    if os.path.splitext(filepath.lower())[1] == ".json":
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("Curated JSON input must be an object")
        structured = payload.get("extracted_data", payload)
        if not isinstance(structured, dict) or not any(
            key in structured for key in ("taxonomy", "biology", "development", "diseases", "nutrition", "behavior", "conservation", "veterinary_notes", "summary")
        ):
            raise ValueError("Curated JSON must contain topic fields or an extracted_data object")
        source_title = source_title or payload.get("source_title", os.path.basename(filepath))
        author = author or payload.get("author", "")
        journal = journal or payload.get("journal", "")
        publication_year = publication_year or str(payload.get("publication_year", ""))
        source_tier = payload.get("source_tier", source_tier)
        save_to_staging(structured, species, category, filepath, source_tier, "",
                        source_title, author, journal, publication_year)
        return 1

    # 1. Load the document
    sections = load_document(filepath)
    if not sections:
        logger.warning("No content extracted from %s", filepath)
        return 0

    # 2. Combine all text
    full_text = "\n\n".join(s["text"] for s in sections)
    filename = os.path.basename(filepath)

    # 3. Handle books (chapter splitting)
    if is_book:
        chapters = detect_chapters(full_text)
        saved = 0
        for chapter in chapters:
            if use_llm:
                structured = structure_with_llm(chapter["text"], llm_model)
                if structured is None:
                    structured = {
                        "common_name": species,
                        "raw_text": chapter["text"][:5000],
                        "summary": f"Chapter: {chapter['title']} from {filename}",
                        "_extraction_failed": True,
                    }
            else:
                structured = {
                    "common_name": species,
                    "raw_text": chapter["text"][:5000],
                    "summary": f"Chapter: {chapter['title']} from {filename}",
                    "_extraction_failed": True,
                }
            save_to_staging(structured, species, category, filepath, source_tier, chapter["title"],
                            source_title, author, journal, publication_year)
            saved += 1
        logger.info("Created %d staging files from book: %s", saved, filename)
        return saved

    # 4. Regular documents — process as a single unit
    if use_llm:
        structured = structure_with_llm(full_text, llm_model)
        if structured is None:
            structured = {
                "common_name": species,
                "raw_text": full_text[:5000],
                "summary": f"Content from {filename} (LLM extraction failed).",
                "_extraction_failed": True,
            }
    else:
        structured = {
            "common_name": species,
            "raw_text": full_text[:5000],
            "summary": f"Content from {filename}.",
            "_extraction_failed": True,
        }

    save_to_staging(structured, species, category, filepath, source_tier, "",
                    source_title, author, journal, publication_year)
    return 1


def main():
    parser = argparse.ArgumentParser(
        description="Process documents (PDF, DOCX, TXT) into staging for vector DB ingestion.",
    )
    parser.add_argument("--file", type=str, help="Path to a single document file.")
    parser.add_argument("--folder", type=str, help="Path to a folder of documents.")
    parser.add_argument("--species", type=str, required=True, help="Animal species name.")
    parser.add_argument(
        "--category", type=str, default="unknown",
        choices=["wild", "domestic", "exotic", "aquatic", "avian", "unknown"],
    )
    parser.add_argument("--source-tier", type=str, default="unverified",
                        choices=["tier_1", "tier_2", "tier_3", "unverified"],
                        help="Credibility tier; defaults to unverified until reviewed.")
    parser.add_argument("--source-title", default="", help="Document or book title.")
    parser.add_argument("--author", default="", help="Author(s), required before approval.")
    parser.add_argument("--journal", default="", help="Journal, publisher, institution, or course.")
    parser.add_argument("--year", dest="publication_year", default="", help="Publication year, required before approval.")
    parser.add_argument("--raw", action="store_true", help="Skip LLM structuring (faster, raw text only).")
    parser.add_argument("--book", action="store_true", help="Treat document as a book (chapter detection).")
    parser.add_argument("--llm-model", type=str, default="llama3.2:3b", help="Ollama model for structuring.")
    args = parser.parse_args()

    if not args.file and not args.folder:
        parser.error("Provide --file or --folder.")

    files = []
    if args.file:
        files.append(args.file)
    if args.folder:
        for root, _, filenames in os.walk(args.folder):
            for fn in sorted(filenames):
                ext = os.path.splitext(fn.lower())[1]
                if ext in SUPPORTED_EXTENSIONS and ext not in SKIP_EXTENSIONS:
                    files.append(os.path.join(root, fn))

    total_staged = 0
    for fp in files:
        count = process_document(
            fp, args.species, args.category, args.source_tier,
            use_llm=not args.raw, is_book=args.book, llm_model=args.llm_model,
            source_title=args.source_title, author=args.author, journal=args.journal,
            publication_year=args.publication_year,
        )
        total_staged += count

    print(f"\n{'='*50}")
    print(f"Document processing complete: {total_staged} staging files created.")
    print(f"Staged files: {os.path.abspath(STAGING_DIR)}")
    print(f"Next: Review and ingest with: python scripts/ingest_staged.py --review")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

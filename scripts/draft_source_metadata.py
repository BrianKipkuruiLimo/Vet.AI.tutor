#!/usr/bin/env python3
"""Draft provenance entries for raw notes that are not yet in the catalog.

Reads each supported file in data/raw_notes/, pulls title/author/year from the
document's embedded properties (falling back to the filename), and writes the
drafts to data/source_metadata.draft.json. It never edits the real catalog.

Every draft has "_needs_review": true and an empty "source_tier", so
rebuild_vectorstore.py refuses it until a person has checked it. To approve a
file: verify the fields, set source_tier, delete "_needs_review" and the
"_detected" notes, then move the entry into data/source_metadata.json.

Usage:
    python scripts/draft_source_metadata.py
    python scripts/draft_source_metadata.py --course "BVM Year 3" --default-author "University of Nairobi"
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from vet_ai_tutor.config import (  # noqa: E402
    RAW_DIR, SKIP_EXTENSIONS, SOURCE_CATALOG_PATH, SUPPORTED_EXTENSIONS, project_path,
)

_YEAR_RE = re.compile(r"(?<!\d)(19[5-9]\d|20[0-4]\d)(?!\d)")
_OFFICE_PREFIX_RE = re.compile(r"^Microsoft (?:PowerPoint|Word|Excel) - ", re.IGNORECASE)
_DR_RE = re.compile(r"\b(?:Dr|Prof)\.?\s+[A-Z][\w.]*(?:\s+[A-Z][\w.]*)?")


def _clean_title(stem: str) -> str:
    title = re.sub(r"[_]+", " ", stem)
    title = re.sub(r"\s*\(\d+\)$", "", title)        # "notes (1)" duplicates
    title = re.sub(r"^\d{6,}[\s_-]*", "", title)     # leading upload timestamps
    return re.sub(r"\s{2,}", " ", title).strip(" -")


def _document_properties(path: Path) -> dict:
    """Best-effort title/author/year from the file's own metadata."""
    ext = path.suffix.lower()
    props = {}
    try:
        if ext == ".pdf":
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(str(path))
            try:
                info = pdf.get_metadata_dict()
            finally:
                pdf.close()
            props = {"title": info.get("Title"), "author": info.get("Author"),
                     "year": str(info.get("CreationDate", ""))[2:6]}
        elif ext == ".docx":
            from docx import Document
            core = Document(str(path)).core_properties
            props = {"title": core.title, "author": core.author,
                     "year": str(core.created.year) if core.created else None}
        elif ext in (".pptx", ".pptm"):
            from pptx import Presentation
            core = Presentation(str(path)).core_properties
            props = {"title": core.title, "author": core.author,
                     "year": str(core.created.year) if core.created else None}
    except Exception as exc:
        props = {"error": f"could not read properties: {exc}"}
    return {k: str(v).strip() for k, v in props.items() if v and str(v).strip()}


def draft_entry(path: Path, rel: str, course: str, default_author: str) -> dict:
    props = _document_properties(path)
    filename_year = _YEAR_RE.findall(path.stem)
    filename_author = _DR_RE.search(path.stem)
    # Office tools often stamp generic titles/authors such as "PowerPoint Presentation" or "user".
    # "Print to PDF" from Office prefixes the app name: "Microsoft PowerPoint - HERNIAS ..."
    title = _OFFICE_PREFIX_RE.sub("", props.get("title", "")).strip()
    if not title or title.lower() in {"powerpoint presentation", "untitled", "microsoft word"}:
        title = _clean_title(path.stem)
    author = (filename_author.group(0) if filename_author else "") or props.get("author", "")
    if author.lower() in {"user", "admin", "owner", "hp", "dell", "lenovo"}:
        author = ""
    year = filename_year[-1] if filename_year else props.get("year", "")
    if not _YEAR_RE.fullmatch(year or ""):
        year = ""
    return {
        "source_title": title,
        "author": author or default_author,
        "journal": course,
        "publication_year": year,
        "source_url": "file://raw_notes/" + quote(rel),
        "source_tier": "",
        "_needs_review": True,
        "_detected": {"properties": props, "filename": path.name},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=project_path(RAW_DIR))
    parser.add_argument("--catalog", default=project_path(SOURCE_CATALOG_PATH))
    parser.add_argument("--output", default=project_path("data/source_metadata.draft.json"))
    parser.add_argument("--course", default="", help="Course or institution for the 'journal' field")
    parser.add_argument("--default-author", default="", help="Used when no author is detected")
    parser.add_argument("--fresh", action="store_true",
                        help="re-draft every uncatalogued file, discarding edits in the existing draft file")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    try:
        catalog = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
    except FileNotFoundError:
        catalog = {}
    names = {}
    files = []
    for root, _, filenames in os.walk(raw_dir):
        for fn in sorted(filenames):
            ext = os.path.splitext(fn.lower())[1]
            if ext in SKIP_EXTENSIONS or ext not in SUPPORTED_EXTENSIONS:
                continue
            path = Path(root) / fn
            files.append(path)
            names[fn] = names.get(fn, 0) + 1

    output = Path(args.output)
    existing = {}
    if output.exists() and not args.fresh:
        existing = json.loads(output.read_text(encoding="utf-8"))
    drafts = {}
    added = 0
    for path in files:
        rel = path.relative_to(raw_dir).as_posix()
        if rel in catalog or path.name in catalog:
            continue
        # Filenames must be unique to be usable as catalog keys; otherwise use the relative path.
        key = path.name if names[path.name] == 1 else rel
        if key in existing:
            drafts[key] = existing[key]  # keep reviewer edits
        else:
            drafts[key] = draft_entry(path, rel, args.course, args.default_author)
            added += 1

    tmp = output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(drafts, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, output)
    missing_year = sum(1 for d in drafts.values() if not d["publication_year"])
    missing_author = sum(1 for d in drafts.values() if not d["author"])
    print(f"Catalogued already: {len(files) - len(drafts)}  |  drafts: {len(drafts)} "
          f"({added} new, {len(drafts) - added} kept)  ->  {output}")
    print(f"Drafts missing a year: {missing_year}; missing an author: {missing_author}")
    print("Review each draft, set source_tier, remove _needs_review/_detected, "
          "then move it into the catalog.")


if __name__ == "__main__":
    main()

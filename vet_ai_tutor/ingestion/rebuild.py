"""Full rebuild of the knowledge index from data/raw_notes/.

Only files with a reviewed catalog entry are indexed; everything else is
reported, so files can be approved in batches. `collect()` does the loading
and validation without embeddings, which makes `--dry-run` cheap.
"""
import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from vet_ai_tutor.config import (
    CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, RAW_DIR, SOURCE_CATALOG_PATH,
    VECTOR_BACKEND, VECTOR_PATH, WHITELIST_PATH, project_path,
)
from vet_ai_tutor.ingestion.indexer import build_index, chunk_documents
from vet_ai_tutor.ingestion.loaders import is_supported, load_file
from vet_ai_tutor.ingestion.provenance import (
    ProvenanceError, catalog_entry, load_catalog, load_whitelist, validate_provenance,
)

SUMMARY_PATH = "archive/rebuild_summary.json"


@dataclass
class RebuildReport:
    indexed_files: List[str] = field(default_factory=list)
    pages: int = 0
    chunks: int = 0
    not_approved: List[dict] = field(default_factory=list)   # no or invalid catalog entry
    unreadable: List[dict] = field(default_factory=list)     # loader errors / legacy formats
    needs_ocr: List[str] = field(default_factory=list)        # mostly scanned PDFs
    empty: List[str] = field(default_factory=list)           # readable but no text
    skipped: int = 0                                          # unsupported file types
    backup_path: Optional[str] = None
    dry_run: bool = False
    embeddings_cached: int = 0     # chunks reused from the embedding cache
    embeddings_computed: int = 0   # chunks embedded during this run

    def summary_lines(self) -> List[str]:
        lines = [
            f"Indexed files:   {len(self.indexed_files)} ({self.pages} pages/slides, {self.chunks} chunks)",
            f"Not approved:    {len(self.not_approved)} (no reviewed catalog entry)",
            f"Unreadable:      {len(self.unreadable)}",
            f"Needs OCR:       {len(self.needs_ocr)} (run scripts/ocr_scanned_pdfs.py)",
            f"No text:         {len(self.empty)}",
            f"Unsupported:     {self.skipped}",
        ]
        if not self.dry_run:
            lines.append(f"Embeddings:      {self.embeddings_computed} computed, "
                         f"{self.embeddings_cached} reused from cache")
        for item in self.unreadable[:10]:
            lines.append(f"  unreadable: {item['file']}: {item['error']}")
        return lines


def collect(raw_dir: str = RAW_DIR, catalog_path: str = SOURCE_CATALOG_PATH,
            whitelist_path: str = WHITELIST_PATH, report: Optional[RebuildReport] = None,
            progress=print):
    """Load every approved file under raw_dir; return (documents, report)."""
    report = report or RebuildReport()
    raw_dir = project_path(raw_dir)
    catalog = load_catalog(catalog_path)
    whitelist = load_whitelist(whitelist_path)
    documents = []
    for root, _, files in os.walk(raw_dir):
        for fn in sorted(files):
            path = os.path.join(root, fn)
            if not is_supported(path):
                report.skipped += 1
                continue
            try:
                stored = validate_provenance(catalog_entry(catalog, raw_dir, path), fn, whitelist)
            except ProvenanceError as exc:
                report.not_approved.append({"file": fn, "reason": str(exc)})
                continue
            try:
                loaded = load_file(path)
            except Exception as exc:  # one bad file must not stop the rebuild
                report.unreadable.append({"file": fn, "error": f"{type(exc).__name__}: {exc}"})
                continue
            if loaded.needs_ocr:
                report.needs_ocr.append(fn)
            if not loaded.documents:
                report.empty.append(fn)
                continue
            for doc in loaded.documents:
                doc.metadata.update(stored)
                doc.metadata["source"] = fn
                doc.metadata["source_path"] = os.path.relpath(path, raw_dir)
            documents.extend(loaded.documents)
            report.indexed_files.append(fn)
            report.pages += len(loaded.documents)
            progress(f"Loaded {fn}: {len(loaded.documents)} pages/slides"
                     + (f" ({'; '.join(loaded.notes)})" if loaded.notes else ""))
    return documents, report


def rebuild(dry_run: bool = False, embeddings=None, raw_dir: str = RAW_DIR,
            catalog_path: str = SOURCE_CATALOG_PATH, whitelist_path: str = WHITELIST_PATH,
            target: str = VECTOR_PATH, summary_path: str = SUMMARY_PATH,
            progress=print) -> RebuildReport:
    """Rebuild the index (or, with dry_run, only report what would be indexed)."""
    report = RebuildReport(dry_run=dry_run)
    documents, report = collect(raw_dir, catalog_path, whitelist_path, report, progress)
    chunks = chunk_documents(documents)
    report.chunks = len(chunks)
    if not dry_run:
        if embeddings is None:
            from vet_ai_tutor.embeddings import load_embeddings
            from vet_ai_tutor.ingestion.embedding_cache import CachedEmbeddings
            base = load_embeddings()
            if base is None:
                raise RuntimeError(f"No embedding backend available; is Ollama running with {EMBEDDING_MODEL}?")
            embeddings = CachedEmbeddings(base)
        report.backup_path = build_index(chunks, embeddings, target, progress=progress)
        report.embeddings_cached = getattr(embeddings, "hits", 0)
        report.embeddings_computed = getattr(embeddings, "misses", 0)
    write_summary(report, summary_path)
    return report


def write_summary(report: RebuildReport, path: str = SUMMARY_PATH) -> str:
    out = project_path(path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({**asdict(report), "vector_path": VECTOR_PATH, "vector_backend": VECTOR_BACKEND,
                   "embedding_model": EMBEDDING_MODEL, "chunk_size": CHUNK_SIZE,
                   "chunk_overlap": CHUNK_OVERLAP}, f, indent=2)
    return out

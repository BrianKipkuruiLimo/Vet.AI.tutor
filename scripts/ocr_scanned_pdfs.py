#!/usr/bin/env python3
"""OCR PDF pages that have no text layer (developer machine, one-off).

Renders each empty page with pypdfium2, runs the Tesseract CLI on it, and
stores the text in data/cache/ocr/<sha256>.json. The loaders read that cache
automatically, so the original PDFs are never modified and the provenance
entry (keyed by filename) stays the same. Already-OCR'd pages are skipped.

Usage:
    python scripts/ocr_scanned_pdfs.py              # PDFs that are mostly scanned
    python scripts/ocr_scanned_pdfs.py --all-empty  # also image-only pages in text PDFs
    python scripts/ocr_scanned_pdfs.py --file "data/raw_notes/CamScanner 11-18-2023 11.07.pdf"

Requires: sudo apt install tesseract-ocr
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from vet_ai_tutor.config import RAW_DIR, project_path  # noqa: E402
from vet_ai_tutor.ingestion.loaders import (  # noqa: E402
    MIN_PAGE_CHARS, OCR_MIN_CONFIDENCE, load_file, ocr_cache_path,
)

# 300 DPI is Tesseract's recommended input resolution (PDF user space is 72 DPI).
RENDER_SCALE = 300 / 72


def parse_tsv(tsv: str) -> tuple[str, float]:
    """Rebuild line text from Tesseract TSV output; return (text, mean word confidence)."""
    lines: dict = {}
    confidences = []
    for row in tsv.splitlines()[1:]:
        cols = row.split("\t")
        if len(cols) < 12 or not cols[11].strip():
            continue
        conf = float(cols[10])
        if conf < 0:
            continue
        confidences.append(conf)
        lines.setdefault((int(cols[2]), int(cols[3]), int(cols[4])), []).append(cols[11])
    text = "\n".join(" ".join(words) for _, words in sorted(lines.items()))
    return text, (sum(confidences) / len(confidences) if confidences else 0.0)


def ocr_pages(pdf_path: str, pages: list[int], lang: str) -> dict:
    import pypdfium2 as pdfium

    tesseract = shutil.which("tesseract")
    if not tesseract:
        raise SystemExit("Tesseract is not installed: sudo apt install tesseract-ocr")
    results = {}
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        with tempfile.TemporaryDirectory(prefix="vet-ocr-") as tmp:
            for number in pages:
                page = pdf[number - 1]
                image_path = os.path.join(tmp, f"page{number}.png")
                page.render(scale=RENDER_SCALE, grayscale=True).to_pil().save(image_path)
                page.close()
                proc = subprocess.run([tesseract, image_path, "stdout", "-l", lang, "tsv"],
                                      capture_output=True, text=True, timeout=300)
                if proc.returncode == 0:
                    text, confidence = parse_tsv(proc.stdout)
                    results[str(number)] = {"text": text, "confidence": round(confidence, 1)}
    finally:
        pdf.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", help="OCR a single PDF")
    parser.add_argument("--raw-dir", default=project_path(RAW_DIR))
    parser.add_argument("--all-empty", action="store_true",
                        help="OCR every empty page, not only PDFs that are mostly scanned")
    parser.add_argument("--lang", default="eng", help="Tesseract language (default: eng)")
    args = parser.parse_args()

    paths = [args.file] if args.file else sorted(str(p) for p in Path(args.raw_dir).rglob("*.pdf"))
    total_pages = 0
    t0 = time.time()
    for path in paths:
        loaded = load_file(path)
        if not loaded.empty_pages or not (args.file or args.all_empty or loaded.needs_ocr):
            continue
        cache_file = ocr_cache_path(path)
        cached = {}
        if os.path.isfile(cache_file):
            with open(cache_file, encoding="utf-8") as f:
                cached = json.load(f).get("pages", {})
        todo = [n for n in loaded.empty_pages if str(n) not in cached]
        if not todo:
            continue
        print(f"OCR {os.path.basename(path)}: {len(todo)} page(s)...", flush=True)
        cached.update(ocr_pages(path, todo, args.lang))
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"source": os.path.basename(path), "engine": "tesseract", "lang": args.lang,
                       "pages": cached}, f, ensure_ascii=False, indent=1)
        good = [n for n in todo if cached.get(str(n), {}).get("confidence", 0) >= OCR_MIN_CONFIDENCE
                and len(cached[str(n)]["text"].strip()) >= MIN_PAGE_CHARS]
        low = [n for n in todo if n not in good]
        print(f"  {len(good)}/{len(todo)} pages usable"
              + (f"; low confidence (handwriting or poor scan), not indexed: {low}" if low else ""), flush=True)
        total_pages += len(todo)
    print(f"Done: {total_pages} page(s) OCR'd in {time.time() - t0:.0f}s. Rebuild the index to include them.")


if __name__ == "__main__":
    main()

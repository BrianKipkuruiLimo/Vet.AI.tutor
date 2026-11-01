"""Document loaders: one LangChain Document per PDF page, slide, or Word file.

All loaders are pure-Python or bundled-binary (no PyTorch, no system tools):

- PDF:  pypdfium2 (BSD-3/Apache-2.0). Keeps line breaks between bullets and is
        ~15x faster than pypdf on lecture slides exported to PDF.
- PPTX: python-pptx, including tables, grouped shapes and speaker notes.
- DOCX: python-docx, paragraphs and tables in document order.

Pages with no text layer are reported in `LoadedFile.empty_pages`. Text from
scripts/ocr_scanned_pdfs.py is read back from the OCR cache for those pages.
Legacy .doc/.ppt files are converted once with LibreOffice into a cache.
Archives python-docx/python-pptx refuse (corrupt media, odd packaging) fall
back to reading the document XML directly.
"""
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import struct
import tempfile
import zipfile
import zlib
from xml.etree import ElementTree
from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document

from vet_ai_tutor.config import (
    INGEST_CACHE_DIR, OCR_MIN_CONFIDENCE as _OCR_MIN_CONFIDENCE,
    SKIP_EXTENSIONS, SUPPORTED_EXTENSIONS, project_path,
)

# A page with less text than this is treated as having no text layer.
MIN_PAGE_CHARS = 30
# A PDF is reported as scanned when at least this share of pages is empty.
SCANNED_PAGE_RATIO = 0.5

CACHE_DIR = INGEST_CACHE_DIR
# Mean Tesseract word confidence (0-100) an OCR'd page needs to be indexed.
# Handwriting typically scores well below this and produces gibberish.
OCR_MIN_CONFIDENCE = _OCR_MIN_CONFIDENCE


class UnsupportedDocument(ValueError):
    """The file type cannot be read; the message says how to convert it."""


@dataclass
class LoadedFile:
    path: str
    kind: str
    documents: List[Document] = field(default_factory=list)
    total_pages: int = 0
    empty_pages: List[int] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)  # e.g. "converted from .ppt", "OCR text used"

    @property
    def needs_ocr(self) -> bool:
        """True when most PDF pages have no extractable text (scanned images)."""
        return (
            self.kind == "pdf" and self.total_pages > 0
            and len(self.empty_pages) / self.total_pages >= SCANNED_PAGE_RATIO
        )


def is_supported(path: str) -> bool:
    ext = os.path.splitext(path.lower())[1]
    return ext in SUPPORTED_EXTENSIONS and ext not in SKIP_EXTENSIONS


def load_file(path: str) -> LoadedFile:
    """Load a supported document. Raises UnsupportedDocument for unreadable formats."""
    ext = os.path.splitext(path.lower())[1]
    kind = SUPPORTED_EXTENSIONS.get(ext)
    if ext in SKIP_EXTENSIONS or kind is None:
        raise UnsupportedDocument(f"Unsupported file type: {ext or 'no extension'}")
    loader = _LOADERS.get(kind)
    if loader is None:
        raise UnsupportedDocument(f"No loader for {kind} files")
    result = loader(path)
    result.kind = kind
    source = os.path.basename(path)
    for doc in result.documents:
        doc.metadata.setdefault("source", source)
    return result


# ---------------------------------------------------------------------------
# Text cleanup
# ---------------------------------------------------------------------------
_SPACES_RE = re.compile(r"[ \t ]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize line endings and whitespace; drop soft hyphens and control chars."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("­", "").replace("￾", "").replace("\x02", "")
    lines = [_SPACES_RE.sub(" ", line).strip() for line in text.split("\n")]
    return _BLANK_LINES_RE.sub("\n\n", "\n".join(lines)).strip()


def _table_rows(rows) -> List[str]:
    """Render table rows as 'cell | cell', skipping empty rows and repeated merged cells."""
    out = []
    for row in rows:
        cells: List[str] = []
        for value in row:
            value = clean_text(value).replace("\n", " ")
            if value and (not cells or cells[-1] != value):
                cells.append(value)
        if cells:
            out.append(" | ".join(cells))
    return out


# ---------------------------------------------------------------------------
# Caches (keyed by file content, so renamed files still hit)
# ---------------------------------------------------------------------------
def file_digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cache_path(kind: str, digest: str, suffix: str) -> str:
    return os.path.join(project_path(CACHE_DIR), kind, digest + suffix)


def ocr_cache_path(path: str) -> str:
    return cache_path("ocr", file_digest(path), ".json")


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def _load_pdf(path: str) -> LoadedFile:
    import pypdfium2 as pdfium

    result = LoadedFile(path=path, kind="pdf")
    pdf = pdfium.PdfDocument(path)
    try:
        result.total_pages = len(pdf)
        texts = {}
        for index in range(len(pdf)):
            page = pdf[index]
            textpage = page.get_textpage()
            try:
                texts[index + 1] = clean_text(textpage.get_text_range())
            finally:
                textpage.close()
                page.close()
    finally:
        pdf.close()

    ocr = {}
    ocr_file = ocr_cache_path(path)
    if os.path.isfile(ocr_file):
        with open(ocr_file, encoding="utf-8") as f:
            ocr = {int(k): v["text"] for k, v in json.load(f).get("pages", {}).items()
                   if v.get("confidence", 0) >= OCR_MIN_CONFIDENCE}
    for number, text in texts.items():
        if len(text) < MIN_PAGE_CHARS and len(ocr.get(number, "")) >= MIN_PAGE_CHARS:
            text = clean_text(ocr[number])
            result.documents.append(Document(page_content=text, metadata={"page": number, "ocr": True}))
        elif len(text) < MIN_PAGE_CHARS:
            result.empty_pages.append(number)
        else:
            result.documents.append(Document(page_content=text, metadata={"page": number}))
    if any(d.metadata.get("ocr") for d in result.documents):
        result.notes.append("OCR text used for pages without a text layer")
    return result


# ---------------------------------------------------------------------------
# Legacy Office formats and damaged archives
# ---------------------------------------------------------------------------
def convert_legacy(path: str, target_ext: str) -> str:
    """Convert .doc/.ppt with LibreOffice (cached); return the converted file path."""
    out = cache_path("converted", file_digest(path), "." + target_ext)
    if os.path.isfile(out):
        return out
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise UnsupportedDocument(
            f"Legacy {os.path.splitext(path)[1]} format needs LibreOffice to convert; "
            f"install it or convert manually: soffice --headless --convert-to {target_ext} <file>"
        )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vet-convert-") as tmp:
        # A private profile dir lets conversion run while LibreOffice is open elsewhere.
        proc = subprocess.run(
            [soffice, f"-env:UserInstallation=file://{tmp}/profile", "--headless",
             "--convert-to", target_ext, "--outdir", tmp, os.path.abspath(path)],
            capture_output=True, text=True, timeout=180,
        )
        produced = [f for f in os.listdir(tmp) if f.endswith("." + target_ext)]
        if not produced:
            hint = ""
            if "could not be loaded" in (proc.stderr + proc.stdout):
                # libreoffice-core alone has no import filters for documents.
                hint = (" LibreOffice is missing its document components; install them with: "
                        "sudo apt install libreoffice-writer-nogui libreoffice-impress-nogui")
            raise UnsupportedDocument(f"LibreOffice could not convert {os.path.basename(path)}.{hint}")
        shutil.move(os.path.join(tmp, produced[0]), out)
    return out


def _is_legacy_ole(path: str) -> bool:
    """True for pre-2007 binary Office files (OLE compound documents)."""
    with open(path, "rb") as f:
        return f.read(8) == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


_A_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _xml_paragraphs(xml: bytes, para_tag: str, text_tag: str) -> List[str]:
    root = ElementTree.fromstring(xml)
    out = []
    for para in root.iter(para_tag):
        text = "".join(t.text or "" for t in para.iter(text_tag))
        if text.strip():
            out.append(text.strip())
    return out


def _slide_number(name: str) -> int:
    match = re.search(r"(\d+)\.xml$", name)
    return int(match.group(1)) if match else 0


def _pptx_from_xml(path: str) -> LoadedFile:
    """Read slide text straight from the archive, ignoring damaged media."""
    result = LoadedFile(path=path, kind="pptx")
    with zipfile.ZipFile(path) as zf:
        slides = sorted((n for n in zf.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                        key=_slide_number)
        result.total_pages = len(slides)
        for name in slides:
            number = _slide_number(name)
            parts = _xml_paragraphs(zf.read(name), _A_NS + "p", _A_NS + "t")
            notes_name = f"ppt/notesSlides/notesSlide{number}.xml"
            if notes_name in zf.namelist():
                notes = [p for p in _xml_paragraphs(zf.read(notes_name), _A_NS + "p", _A_NS + "t")
                         if not p.isdigit()]  # drop the slide-number placeholder
                if notes:
                    parts.append("Speaker notes: " + " ".join(notes))
            text = clean_text("\n".join(parts))
            if len(text) < MIN_PAGE_CHARS:
                result.empty_pages.append(number)
            else:
                result.documents.append(Document(page_content=text, metadata={"slide": number}))
    result.notes.append("read from slide XML (archive could not be opened normally)")
    return result


def read_damaged_zip_member(path: str, member: str) -> bytes:
    """Recover one member from a truncated ZIP by scanning its local file headers.

    A ZIP whose central directory (at the end of the file) is missing can't be
    opened by zipfile, but each entry still starts with a local header followed
    by its (usually deflate-compressed) data.
    """
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    while (pos := data.find(b"PK\x03\x04", pos)) != -1:
        if pos + 30 > len(data):
            break
        _, _, _, method, _, _, _, comp_size, _, name_len, extra_len = struct.unpack(
            "<IHHHHHIIIHH", data[pos:pos + 30])
        name = data[pos + 30:pos + 30 + name_len].decode("utf-8", "replace")
        start = pos + 30 + name_len + extra_len
        if name == member:
            if method == 0:
                return data[start:start + comp_size]
            if method == 8:
                # Sizes may be deferred to a data descriptor, so decompress until the stream ends.
                return zlib.decompressobj(-15).decompress(data[start:])
            raise ValueError(f"Unsupported compression method {method} for {member}")
        pos = start
    raise KeyError(f"{member} not found in damaged archive {os.path.basename(path)}")


def _docx_from_xml(path: str) -> LoadedFile:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            xml = zf.read("word/document.xml")
    else:
        xml = read_damaged_zip_member(path, "word/document.xml")
    parts = _xml_paragraphs(xml, _W_NS + "p", _W_NS + "t")
    text = clean_text("\n".join(parts))
    result = LoadedFile(path=path, kind="docx", total_pages=1,
                        notes=["read from document XML (archive could not be opened normally)"])
    if len(text) >= MIN_PAGE_CHARS:
        result.documents.append(Document(page_content=text, metadata={}))
    else:
        result.empty_pages.append(1)
    return result


# ---------------------------------------------------------------------------
# PowerPoint
# ---------------------------------------------------------------------------
def _shape_texts(shapes) -> List[str]:
    """Text from shapes in reading order, descending into groups and tables."""
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    parts: List[str] = []
    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            parts.extend(_shape_texts(shape.shapes))
        elif getattr(shape, "has_table", False) and shape.has_table:
            parts.extend(_table_rows(
                [cell.text for cell in row.cells] for row in shape.table.rows
            ))
        elif getattr(shape, "has_text_frame", False) and shape.has_text_frame:
            text = clean_text(shape.text_frame.text)
            if text:
                parts.append(text)
    return parts


def _load_pptx(path: str) -> LoadedFile:
    from pptx import Presentation

    source = path
    if _is_legacy_ole(path):
        source = convert_legacy(path, "pptx")
    try:
        prs = Presentation(source)
    except Exception:
        if not zipfile.is_zipfile(source):
            raise
        result = _pptx_from_xml(source)
        result.path = path
        return result
    result = LoadedFile(path=path, kind="pptx", total_pages=len(prs.slides))
    if source != path:
        result.notes.append("converted from legacy .ppt with LibreOffice")
    for number, slide in enumerate(prs.slides, start=1):
        parts = _shape_texts(slide.shapes)
        notes_frame = slide.notes_slide.notes_text_frame if slide.has_notes_slide else None
        if notes_frame is not None:
            notes = clean_text(notes_frame.text)
            if notes:
                parts.append(f"Speaker notes: {notes}")
        text = "\n".join(parts)
        if len(text) < MIN_PAGE_CHARS:
            result.empty_pages.append(number)
            continue
        result.documents.append(Document(page_content=text, metadata={"slide": number}))
    return result


# ---------------------------------------------------------------------------
# Word
# ---------------------------------------------------------------------------
def _load_docx(path: str) -> LoadedFile:
    from docx import Document as DocxDocument
    from docx.table import Table

    source = path
    if _is_legacy_ole(path):
        source = convert_legacy(path, "docx")
    try:
        doc = DocxDocument(source)
    except Exception:
        result = _docx_from_xml(source)
        result.path = path
        return result
    parts: List[str] = []
    for block in doc.iter_inner_content():
        if isinstance(block, Table):
            parts.extend(_table_rows([cell.text for cell in row.cells] for row in block.rows))
        else:
            text = clean_text(block.text)
            if text:
                parts.append(text)
    text = "\n".join(parts)
    result = LoadedFile(path=path, kind="docx", total_pages=1)
    if source != path:
        result.notes.append("converted from legacy .doc with LibreOffice")
    if len(text) >= MIN_PAGE_CHARS:
        result.documents.append(Document(page_content=text, metadata={}))
    else:
        result.empty_pages.append(1)
    return result


# ---------------------------------------------------------------------------
# Plain text, Markdown, JSON, CSV
# ---------------------------------------------------------------------------
def _load_text(path: str) -> LoadedFile:
    with open(path, encoding="utf-8", errors="replace") as f:
        text = clean_text(f.read())
    result = LoadedFile(path=path, kind="txt", total_pages=1)
    if len(text) >= MIN_PAGE_CHARS:
        result.documents.append(Document(page_content=text, metadata={}))
    return result


def _load_csv(path: str) -> LoadedFile:
    """One document per row, as 'column: value' lines."""
    result = LoadedFile(path=path, kind="csv")
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for number, row in enumerate(csv.DictReader(f), start=1):
            result.total_pages += 1
            text = "\n".join(f"{k}: {v}" for k, v in row.items() if k and v and str(v).strip())
            if len(text) >= 10:
                result.documents.append(Document(page_content=text, metadata={"row": number}))
    return result


_LOADERS = {
    "pdf": _load_pdf,
    "pptx": _load_pptx,
    "ppt": _load_pptx,
    "docx": _load_docx,
    "doc": _load_docx,
    "txt": _load_text,
    "markdown": _load_text,
    "json": _load_text,
    "csv": _load_csv,
}

"""Tests for vet_ai_tutor.ingestion: loaders, provenance, rebuild and indexing.

Fixture documents (PDF, PPTX, DOCX) are generated inside the tests.
"""
import importlib.util
import json
import os
from pathlib import Path

import pytest

from vet_ai_tutor.ingestion import loaders
from vet_ai_tutor.ingestion.loaders import UnsupportedDocument, load_file
from vet_ai_tutor.ingestion.provenance import ProvenanceError, validate_provenance

ROOT = Path(__file__).resolve().parents[1]
WHITELIST = {"merckvetmanual.com": "tier_1", "vet.cornell.edu": "tier_2"}


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Write a minimal valid PDF with one Helvetica text line per page ('' = blank page)."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None,
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET" if text else ""
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        content_ref = len(objects)
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                       f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_ref} 0 R >>")
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)
    return path


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(loaders, "CACHE_DIR", str(tmp_path / "cache"))


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def test_pdf_pages_and_empty_page_detection(tmp_path):
    pdf = make_pdf(tmp_path / "notes.pdf", [
        "Milk fever is hypocalcaemia in dairy cows around calving.", "",
        "Treatment is slow intravenous calcium borogluconate.",
    ])
    loaded = load_file(str(pdf))
    assert [d.metadata["page"] for d in loaded.documents] == [1, 3]
    assert "hypocalcaemia" in loaded.documents[0].page_content
    assert loaded.documents[0].metadata["source"] == "notes.pdf"
    assert loaded.empty_pages == [2]
    assert loaded.total_pages == 3 and not loaded.needs_ocr


def test_scanned_pdf_is_flagged_and_uses_confident_ocr_only(tmp_path):
    pdf = make_pdf(tmp_path / "scan.pdf", ["", ""])
    assert load_file(str(pdf)).needs_ocr
    cache = Path(loaders.ocr_cache_path(str(pdf)))
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"pages": {
        "1": {"text": "Babesia bigemina is transmitted by Rhipicephalus ticks.", "confidence": 91.0},
        "2": {"text": "Tlie qvick hnown f0x jurnps ovcr the lazv dog 1lll", "confidence": 41.0},
    }}), encoding="utf-8")
    loaded = load_file(str(pdf))
    assert len(loaded.documents) == 1
    assert loaded.documents[0].metadata == {"page": 1, "ocr": True, "source": "scan.pdf"}
    assert loaded.empty_pages == [2]  # low-confidence OCR is never indexed


def test_pptx_extracts_tables_groups_and_speaker_notes(tmp_path):
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "Tick-borne diseases of cattle"
    table = slide.shapes.add_table(2, 2, Inches(1), Inches(2), Inches(6), Inches(1)).table
    for (r, c), value in {(0, 0): "Disease", (0, 1): "Vector", (1, 0): "Babesiosis",
                          (1, 1): "Rhipicephalus"}.items():
        table.cell(r, c).text = value
    group = slide.shapes.add_group_shape()
    group.shapes.add_textbox(Inches(1), Inches(4), Inches(4), Inches(1)).text_frame.text = "Grouped caption text"
    slide.notes_slide.notes_text_frame.text = "Emphasize imidocarb dosing."
    blank = prs.slides.add_slide(prs.slide_layouts[6])
    assert blank is not None
    path = tmp_path / "ticks.pptx"
    prs.save(path)

    loaded = load_file(str(path))
    assert loaded.total_pages == 2 and loaded.empty_pages == [2]
    text = loaded.documents[0].page_content
    assert loaded.documents[0].metadata["slide"] == 1
    assert "Tick-borne diseases of cattle" in text
    assert "Babesiosis | Rhipicephalus" in text
    assert "Grouped caption text" in text
    assert "Speaker notes: Emphasize imidocarb dosing." in text


def test_docx_keeps_tables_in_document_order(tmp_path):
    from docx import Document

    doc = Document()
    doc.add_heading("Anthelmintics", level=1)
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Drug", "Dose"
    table.cell(1, 0).text, table.cell(1, 1).text = "Albendazole", "7.5 mg/kg"
    doc.add_paragraph("Withdrawal periods apply to meat and milk.")
    path = tmp_path / "drugs.docx"
    doc.save(path)

    text = load_file(str(path)).documents[0].page_content
    assert text.index("Anthelmintics") < text.index("Albendazole | 7.5 mg/kg") < text.index("Withdrawal")


def test_truncated_docx_is_recovered_from_local_headers(tmp_path):
    from docx import Document

    doc = Document()
    doc.add_paragraph("Veterinary economics and business management lecture notes.")
    path = tmp_path / "damaged.docx"
    doc.save(path)
    data = path.read_bytes()
    path.write_bytes(data[:data.index(b"PK\x01\x02")])  # drop the central directory

    loaded = load_file(str(path))
    assert "business management" in loaded.documents[0].page_content
    assert any("document XML" in note for note in loaded.notes)


def test_legacy_office_without_libreoffice_explains_fix(tmp_path, monkeypatch):
    path = tmp_path / "old.doc"
    path.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 512)
    monkeypatch.setattr(loaders.shutil, "which", lambda name: None)
    with pytest.raises(UnsupportedDocument, match="LibreOffice"):
        load_file(str(path))


def test_unsupported_and_csv(tmp_path):
    with pytest.raises(UnsupportedDocument):
        load_file(str(tmp_path / "photo.jpg"))
    csv_path = tmp_path / "doses.csv"
    csv_path.write_text("drug,dose\nOxytetracycline,10 mg/kg\n,\n", encoding="utf-8")
    docs = load_file(str(csv_path)).documents
    assert len(docs) == 1 and docs[0].page_content == "drug: Oxytetracycline\ndose: 10 mg/kg"


def test_clean_text_normalizes_whitespace():
    assert loaders.clean_text("Line\r\none   two­\n\n\n\nnext") == "Line\none two\n\nnext"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _entry(**overrides):
    return {"source_title": "Ruminant Medicine", "author": "Dr. A", "journal": "BVM",
            "publication_year": "2022", "source_url": "file://raw_notes/a.pdf",
            "source_tier": "tier_1", **overrides}


@pytest.mark.parametrize("overrides, message", [
    ({"_needs_review": True}, "draft"),
    ({"author": ""}, "missing author"),
    ({"publication_year": "22"}, "publication_year"),
    ({"source_tier": "tier_3"}, "tier_1 and tier_2"),
    ({"source_url": "http://merckvetmanual.com/x"}, "https:// or file://"),
    ({"source_url": "https://vet.cornell.edu/x", "source_tier": "tier_1"}, "not whitelisted"),
    ({"source_url": "https://random-blog.example/x"}, "not whitelisted"),
])
def test_provenance_rejections(overrides, message):
    with pytest.raises(ProvenanceError, match=message):
        validate_provenance(_entry(**overrides), "a.pdf", WHITELIST)


def test_provenance_accepts_reviewed_entry():
    stored = validate_provenance(_entry(source_url="https://www.merckvetmanual.com/x"), "a.pdf", WHITELIST)
    assert stored["source_domain"] == "www.merckvetmanual.com"
    assert stored["publication_year"] == "2022"
    with pytest.raises(ProvenanceError, match="no reviewed provenance"):
        validate_provenance(None, "a.pdf", WHITELIST)


# ---------------------------------------------------------------------------
# Rebuild and indexing
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_notes(tmp_path):
    raw = tmp_path / "raw_notes"
    raw.mkdir()
    (raw / "approved.txt").write_text("Mastitis is inflammation of the mammary gland. " * 30, encoding="utf-8")
    (raw / "draft.txt").write_text("Draft content that must not be indexed yet. " * 5, encoding="utf-8")
    (raw / "photo.jpg").write_bytes(b"\xff\xd8")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({
        "approved.txt": _entry(),
        "draft.txt": _entry(_needs_review=True),
    }), encoding="utf-8")
    return raw, catalog


def test_rebuild_dry_run_reports_without_embedding(raw_notes, tmp_path):
    from vet_ai_tutor.ingestion.rebuild import rebuild

    raw, catalog = raw_notes
    summary = tmp_path / "summary.json"
    report = rebuild(dry_run=True, raw_dir=str(raw), catalog_path=str(catalog),
                     whitelist_path=str(ROOT / "data" / "whitelist.json"),
                     summary_path=str(summary), progress=lambda _m: None)
    assert report.indexed_files == ["approved.txt"]
    assert [item["file"] for item in report.not_approved] == ["draft.txt"]
    assert report.skipped == 1
    assert report.chunks > 1
    assert json.loads(summary.read_text())["indexed_files"] == ["approved.txt"]


def test_collect_attaches_provenance_to_every_document(raw_notes):
    from vet_ai_tutor.ingestion.rebuild import collect

    raw, catalog = raw_notes
    docs, _ = collect(str(raw), str(catalog), str(ROOT / "data" / "whitelist.json"), progress=lambda _m: None)
    assert docs and all(d.metadata["source_tier"] == "tier_1" and d.metadata["author"] == "Dr. A"
                        and d.metadata["source"] == "approved.txt" for d in docs)


def test_build_index_swaps_atomically_and_keeps_backup(tmp_path):
    from langchain_core.documents import Document
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from vet_ai_tutor.embeddings import read_index_info
    from vet_ai_tutor.ingestion.indexer import build_index, chunk_documents

    target = tmp_path / "index"
    embeddings = DeterministicFakeEmbedding(size=8)
    chunks = chunk_documents([Document(page_content="Bovine babesiosis notes. " * 50, metadata={"source": "a"})])
    assert [c.metadata["chunk"] for c in chunks] == list(range(len(chunks)))

    assert build_index(chunks, embeddings, str(target), str(tmp_path / "backups"), progress=lambda _m: None) is None
    assert read_index_info(str(target))["embedding_model"]
    backup = build_index(chunks[:1], embeddings, str(target), str(tmp_path / "backups"), progress=lambda _m: None)
    assert backup and os.path.isdir(backup)
    assert not any(p.name.startswith(".vet-index-rebuild-") for p in tmp_path.iterdir())
    with pytest.raises(ValueError, match="No chunks"):
        build_index([], embeddings, str(target), str(tmp_path / "backups"))


# ---------------------------------------------------------------------------
# OCR script helpers
# ---------------------------------------------------------------------------

def test_ocr_tsv_parsing_and_confidence():
    spec = importlib.util.spec_from_file_location("ocr", ROOT / "scripts" / "ocr_scanned_pdfs.py")
    ocr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ocr)
    header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
    rows = [
        "5\t1\t1\t1\t1\t1\t0\t0\t1\t1\t90\tMilk",
        "5\t1\t1\t1\t1\t2\t0\t0\t1\t1\t80\tfever",
        "4\t1\t1\t1\t2\t0\t0\t0\t1\t1\t-1\t",
        "5\t1\t1\t1\t2\t1\t0\t0\t1\t1\t70\tcalcium",
    ]
    text, confidence = ocr.parse_tsv("\n".join([header, *rows]))
    assert text == "Milk fever\ncalcium"
    assert confidence == 80.0


def test_embedding_cache_reuses_vectors(tmp_path):
    from vet_ai_tutor.ingestion.embedding_cache import CachedEmbeddings

    class CountingEmbeddings:
        calls = 0

        def embed_documents(self, texts):
            CountingEmbeddings.calls += len(texts)
            return [[float(len(t)), 0.5] for t in texts]

        def embed_query(self, text):
            return [1.0, 0.0]

    path = str(tmp_path / "cache.sqlite")
    cache = CachedEmbeddings(CountingEmbeddings(), path)
    first = cache.embed_documents(["alpha", "beta"])
    cache.close()

    reopened = CachedEmbeddings(CountingEmbeddings(), path)  # survives a restart
    again = reopened.embed_documents(["beta", "gamma", "alpha"])
    assert CountingEmbeddings.calls == 3  # only "gamma" was new
    assert again[0] == first[1] and again[2] == first[0]
    assert (reopened.hits, reopened.misses) == (2, 1)

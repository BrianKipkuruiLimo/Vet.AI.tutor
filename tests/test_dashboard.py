"""Headless UI tests for the developer dashboard (streamlit.testing AppTest)."""
import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.test_devtools import _draft, dev_paths  # noqa: E402,F401  (fixture)
from vet_ai_tutor import quiz  # noqa: E402
from vet_ai_tutor.devtools import catalog  # noqa: E402

ENTRY = str(Path(__file__).resolve().parents[1] / "dev_dashboard.py")
PAGES = ["app_pages/sources.py", "app_pages/index.py", "app_pages/staging.py",
         "app_pages/evaluation.py", "app_pages/quiz.py", "app_pages/release.py"]


def open_page(page: str) -> AppTest:
    at = AppTest.from_file(ENTRY, default_timeout=30).run()
    if page != PAGES[0]:
        at.switch_page(page).run()
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_on_an_empty_project(dev_paths, page):  # noqa: F811
    at = open_page(page)
    assert at.title[0].value  # the entry point titles each page
    assert len(at.metric) >= 5  # summary row


def test_sources_page_approves_ready_drafts(dev_paths):  # noqa: F811
    for name in ("ready.pdf", "blocked.pdf"):
        (Path(dev_paths.raw_dir) / name).write_bytes(b"%PDF")
    catalog.write_json(dev_paths.drafts, {"ready.pdf": _draft(source_tier="tier_1"),
                                          "blocked.pdf": _draft(source_tier="tier_1", publication_year="")})
    at = open_page(PAGES[0])
    approve = next(b for b in at.button if b.label.startswith("Approve 1 ready"))
    approve.click().run()
    assert not at.exception
    assert list(catalog.read_json(dev_paths.catalog)) == ["ready.pdf"]
    assert list(catalog.read_json(dev_paths.drafts)) == ["blocked.pdf"]


def test_sources_page_bulk_sets_a_field(dev_paths):  # noqa: F811
    (Path(dev_paths.raw_dir) / "a.pdf").write_bytes(b"%PDF")
    catalog.write_json(dev_paths.drafts, {"a.pdf": _draft()})
    at = open_page(PAGES[0])
    at.selectbox(key="bulk_field").select("journal").run()
    at.text_input(key="bulk_value").set_value("VMED 344 lectures").run()
    next(b for b in at.button if b.label.startswith("Apply to 1")).click().run()
    assert not at.exception
    assert catalog.read_json(dev_paths.drafts)["a.pdf"]["journal"] == "VMED 344 lectures"


def test_index_page_shows_last_report(dev_paths):  # noqa: F811
    Path(dev_paths.rebuild_summary).write_text(json.dumps({
        "indexed_files": ["a.pdf"], "chunks": 12, "not_approved": [], "dry_run": True,
        "unreadable": [{"file": "old.doc", "error": "needs LibreOffice"}], "needs_ocr": ["scan.pdf"], "empty": []}))
    at = open_page(PAGES[1])
    assert any(m.label == "Unreadable" and m.value == "1" for m in at.metric)
    assert any("dry run" in s.value for s in at.subheader)


def test_quiz_page_review_and_publish(dev_paths):  # noqa: F811
    from tests.test_quiz import entry
    pending = Path(dev_paths.quiz_pending)
    pending.parent.mkdir(parents=True)
    pending.write_text(json.dumps([entry(status="pending_review")]))
    at = open_page(PAGES[4])
    at.button(key="quiz_approve").click().run()
    assert json.loads(pending.read_text())[0]["status"] == "approved"
    next(b for b in at.button if b.label == "Publish approved questions").click().run()
    assert not at.exception
    assert len(quiz.QuizBank(quiz.bank_path(dev_paths.index))) == 1


def test_release_page_generates_signing_key(dev_paths):  # noqa: F811
    at = open_page(PAGES[5])
    next(b for b in at.button if b.label == "Generate signing key").click().run()
    assert not at.exception
    assert Path(dev_paths.signing_key).exists() and Path(dev_paths.public_key).exists()
    at.run()
    assert not any(b.label == "Generate signing key" for b in at.button)


def test_staging_page_lists_records(dev_paths):  # noqa: F811
    staging = Path(dev_paths.staging)
    staging.mkdir()
    (staging / "giraffe_1.json").write_text(json.dumps({
        "status": "pending_review", "species": "giraffe", "source_title": "Giraffe care",
        "source_tier": "tier_2", "source_url": "https://vet.cornell.edu/x"}))
    at = open_page(PAGES[2])
    assert any("1 of 1 records" in c.value for c in at.caption)

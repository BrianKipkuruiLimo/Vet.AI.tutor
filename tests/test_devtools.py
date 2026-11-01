"""Tests for the dashboard logic in vet_ai_tutor.devtools (no Streamlit)."""
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from vet_ai_tutor.devtools import catalog, jobs, paths, status

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dev_paths(tmp_path, monkeypatch):
    """Point every dashboard path at a temporary project."""
    raw = tmp_path / "raw_notes"
    raw.mkdir()
    real = paths.get_paths()
    p = replace(real, raw_dir=str(raw), catalog=str(tmp_path / "catalog.json"),
                drafts=str(tmp_path / "drafts.json"), whitelist=str(ROOT / "data" / "whitelist.json"),
                staging=str(tmp_path / "staging"), index=str(tmp_path / "index"),
                eval_dir=str(tmp_path / "eval"), qa_pairs=str(tmp_path / "eval" / "qa.json"),
                quiz_pending=str(tmp_path / "quiz" / "pending.json"), jobs=str(tmp_path / "jobs"),
                rebuild_summary=str(tmp_path / "summary.json"), patches=str(tmp_path / "patches"),
                public_key=str(tmp_path / "keys" / "public.pem"),
                signing_key=str(tmp_path / "keys" / "private.pem"), backups=str(tmp_path / "backups"))
    monkeypatch.setattr(paths, "get_paths", lambda: p)
    return p


def wait_for(job_id, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = jobs.get_job(job_id)
        if job and not job.active:
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def test_job_runs_in_background_and_records_result(dev_paths):
    job = jobs.start_job("evaluate_retrieval", command=[sys.executable, "-c", "print('recall 0.5'); print('done')"])
    assert job.status == "running" and job.pid
    finished = wait_for(job.id)
    assert finished.status == "succeeded" and finished.returncode == 0 and finished.finished
    assert jobs.last_line(finished) == "done"
    assert "recall 0.5" in jobs.read_log(finished)


def test_failed_job_is_reported(dev_paths):
    job = jobs.start_job("benchmark", command=[sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"])
    finished = wait_for(job.id)
    assert finished.status == "failed" and finished.returncode == 3


def test_jobs_writing_the_index_are_exclusive_and_cancellable(dev_paths):
    slow = [sys.executable, "-c", "import time; print('working', flush=True); time.sleep(60)"]
    rebuild = jobs.start_job("rebuild", command=slow)
    with pytest.raises(jobs.JobConflict):
        jobs.start_job("ingest_staged", command=slow)   # same "index" group
    with pytest.raises(jobs.JobConflict):
        jobs.start_job("rebuild", command=slow)         # same kind
    other = jobs.start_job("ocr", command=[sys.executable, "-c", "print('ok')"])  # unrelated kind
    assert wait_for(other.id).status == "succeeded"
    time.sleep(0.5)
    assert jobs.cancel_job(rebuild.id)
    assert wait_for(rebuild.id).status == "cancelled"
    assert not jobs.cancel_job(rebuild.id)  # already finished


def test_lost_runner_is_marked_failed(dev_paths):
    os.makedirs(dev_paths.jobs)
    job = jobs.Job(id="20260101-000000-abcdef", kind="rebuild", label="Rebuild index", argv=[],
                   status="running", pid=2**22 + 12345, started="2026-01-01T00:00:00+00:00",
                   log_path=os.path.join(dev_paths.jobs, "x.log"))
    jobs._write(job)
    listed = jobs.list_jobs()
    assert listed[0].status == "failed" and "stopped without reporting" in listed[0].meta["error"]
    assert jobs.active_conflict("rebuild") is None


def test_unknown_job_kind_is_rejected(dev_paths):
    with pytest.raises(ValueError):
        jobs.start_job("format_disk")


# ---------------------------------------------------------------------------
# Catalog review
# ---------------------------------------------------------------------------

def _draft(**overrides):
    return {"source_title": "Diseases of foals", "author": "Dr. Kirui G", "journal": "BVM",
            "publication_year": "2022", "source_url": "file://raw_notes/a.pdf", "source_tier": "",
            "_needs_review": True, "_detected": {"filename": "a.pdf"}, **overrides}


@pytest.mark.parametrize("name, code", [
    ("26. VMED 344 Med I Sem II-Equine- Diseases of Foals-Dr. Kirui G.pdf", "VMED 344"),
    ("CAT 1 ANSCI 365.docx", "ANSCI 365"),
    ("VMED-230-IMMUNOLOGY NOTES-2021.doc", "VMED 230"),
    ("Coenurosis-Oestrus Ovis-Tick Paralysis -BVMS3.pptx", "BVMS 3"),
    ("CamScanner 11-18-2023 11.07.pdf", ""),
    ("DRAFT KASEP 1 07.12.2021.pdf", ""),
])
def test_course_code(name, code):
    assert catalog.course_code(name) == code


def test_review_rows_and_approval_flow(dev_paths):
    for name in ("a.pdf", "b.pdf", "c.pdf"):
        (Path(dev_paths.raw_dir) / name).write_bytes(b"%PDF")
    catalog.write_json(dev_paths.drafts, {"a.pdf": _draft(), "b.pdf": _draft(author="")})
    rows = {r["key"]: r for r in catalog.review_rows(dev_paths.raw_dir, dev_paths.catalog,
                                                     dev_paths.drafts, dev_paths.whitelist)}
    assert rows["a.pdf"]["status"] == "draft" and "tier" in rows["a.pdf"]["problem"]
    assert rows["c.pdf"]["status"] == "no draft"

    assert catalog.bulk_set(dev_paths.drafts, ["a.pdf", "b.pdf"], "source_tier", "tier_1") == 2
    approved, rejected = catalog.approve(["a.pdf", "b.pdf"], dev_paths.drafts, dev_paths.catalog,
                                         dev_paths.whitelist, dev_paths.backups)
    assert approved == ["a.pdf"] and "author" in rejected["b.pdf"]
    saved = catalog.read_json(dev_paths.catalog)["a.pdf"]
    assert "_needs_review" not in saved and "_detected" not in saved and saved["reviewed_at"]
    assert "a.pdf" not in catalog.read_json(dev_paths.drafts)

    catalog.apply_edits(dev_paths.drafts, {"b.pdf": {"author": "Dr. Were", "not_a_field": "x"}})
    assert catalog.approve(["b.pdf"], dev_paths.drafts, dev_paths.catalog, dev_paths.whitelist,
                           dev_paths.backups)[0] == ["b.pdf"]
    assert len(os.listdir(os.path.join(dev_paths.backups, "catalog"))) == 1  # backup before 2nd write

    assert catalog.unapprove(["a.pdf"], dev_paths.drafts, dev_paths.catalog, dev_paths.backups) == ["a.pdf"]
    assert catalog.read_json(dev_paths.drafts)["a.pdf"]["_needs_review"] is True
    with pytest.raises(ValueError):
        catalog.bulk_set(dev_paths.drafts, ["a.pdf"], "status", "approved")


def test_invalid_catalog_entries_are_flagged(dev_paths):
    catalog.write_json(dev_paths.catalog, {"x.pdf": _draft(source_tier="tier_3", _needs_review=False)})
    rows = catalog.review_rows(dev_paths.raw_dir, dev_paths.catalog, dev_paths.drafts, dev_paths.whitelist)
    assert rows[0]["status"] == "invalid" and not rows[0]["file_exists"]


def test_upload_saves_file_and_drafts_metadata(dev_paths):
    key = catalog.add_upload("VMED 344 notes 2023.txt", b"Equine colic notes " * 20, dev_paths.raw_dir,
                             dev_paths.drafts, course="BVM Year 3")
    assert key == "VMED 344 notes 2023.txt"
    assert (Path(dev_paths.raw_dir) / "uploaded" / key).exists()
    draft = catalog.read_json(dev_paths.drafts)[key]
    assert draft["journal"] == "BVM Year 3" and draft["publication_year"] == "2023" and draft["_needs_review"]
    # Same content again: no duplicate. Different content, same name: kept side by side.
    assert catalog.add_upload("VMED 344 notes 2023.txt", b"Equine colic notes " * 20, dev_paths.raw_dir,
                              dev_paths.drafts) == key
    other = catalog.add_upload("VMED 344 notes 2023.txt", b"different", dev_paths.raw_dir, dev_paths.drafts)
    assert other != key and len(os.listdir(Path(dev_paths.raw_dir) / "uploaded")) == 2
    with pytest.raises(ValueError, match="Unsupported"):
        catalog.add_upload("virus.exe", b"x", dev_paths.raw_dir, dev_paths.drafts)


def test_drafter_keeps_existing_drafts(dev_paths, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("drafter", ROOT / "scripts" / "draft_source_metadata.py")
    drafter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drafter)
    for name in ("one.txt", "two.txt"):
        (Path(dev_paths.raw_dir) / name).write_text("text " * 20, encoding="utf-8")
    catalog.write_json(dev_paths.drafts, {"one.txt": _draft(author="Edited by reviewer")})
    monkeypatch.setattr(sys, "argv", ["drafter", "--raw-dir", dev_paths.raw_dir, "--catalog",
                                      dev_paths.catalog, "--output", dev_paths.drafts])
    drafter.main()
    drafts = catalog.read_json(dev_paths.drafts)
    assert drafts["one.txt"]["author"] == "Edited by reviewer"  # edits survive a re-run
    assert "two.txt" in drafts


# ---------------------------------------------------------------------------
# Status readers
# ---------------------------------------------------------------------------

def test_history_readers(dev_paths):
    os.makedirs(dev_paths.eval_dir)
    Path(dev_paths.eval_dir, "results_hybrid_k5_1780000000.json").write_text(json.dumps({
        "config": {"mode": "hybrid", "k": 5, "chunks": 100},
        "summary": {"total": 10, "recall_at_k": 0.6, "mrr": 0.45}}))
    Path(dev_paths.eval_dir, "results_k4_1771692789.json").write_text("[]")  # old format: ignored
    Path(dev_paths.eval_dir, "benchmark_chroma_1780000100.json").write_text(json.dumps({
        "system_specs": {"llm_model": "qwen3:1.7b"},
        "summary": {"total_questions": 5, "avg_time_per_question": 40.2},
        "answer_checks": {"mean_grounding": 0.8, "citation_rate": 0.9}}))
    retrieval = status.retrieval_history(dev_paths.eval_dir)
    assert len(retrieval) == 1 and retrieval[0]["recall_at_k"] == 0.6
    bench = status.benchmark_history(dev_paths.eval_dir)
    assert bench[0]["grounding"] == 0.8 and bench[0]["faithfulness"] is None


def test_release_status_and_fingerprint(dev_paths):
    from vet_ai_tutor.signing import generate_keypair
    assert status.release_status(dev_paths)["fingerprint"] is None
    generate_keypair(dev_paths.signing_key, dev_paths.public_key)
    release = status.release_status(dev_paths)
    assert release["private_key"] and release["public_key"]
    assert len(release["fingerprint"].split(":")) == 4


def test_qa_and_quiz_stats(dev_paths):
    os.makedirs(dev_paths.eval_dir)
    Path(dev_paths.qa_pairs).write_text(json.dumps([{"source": "a.pdf"}, {"retrieval_missed": True}, {}]))
    assert status.qa_stats(dev_paths.qa_pairs) == {"questions": 3, "with_source": 1, "retrieval_missed": 1}
    assert status.quiz_stats(dev_paths.quiz_pending, "/nonexistent") == {"drafts": {}, "published": 0}

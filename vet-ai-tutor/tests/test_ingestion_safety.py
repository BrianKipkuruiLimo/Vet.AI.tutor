import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_whitelist_fails_closed_and_rejects_http(tmp_path):
    harvester = load_script("harvester")
    with pytest.raises(RuntimeError):
        harvester.load_whitelist(str(tmp_path / "missing.json"))
    whitelist = {"vet.cornell.edu": "tier_2"}
    assert harvester.check_domain("https://vet.cornell.edu/page", whitelist) == "tier_2"
    assert harvester.check_domain("https://fakevet.cornell.edu/page", whitelist) is None
    assert harvester.check_domain("http://vet.cornell.edu/page", whitelist) is None


def test_pending_staging_record_cannot_be_ingested(tmp_path):
    ingestor = load_script("ingest_staged")
    staged = tmp_path / "pending.json"
    staged.write_text(json.dumps({"status": "pending_review"}), encoding="utf-8")
    db, count = ingestor.ingest_file(str(staged), embeddings=None, db=object())
    assert count == 0
    assert staged.exists()


def test_approved_record_requires_complete_provenance(tmp_path):
    ingestor = load_script("ingest_staged")
    staged = tmp_path / "approved.json"
    staged.write_text(json.dumps({"status": "approved", "extracted_data": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete provenance"):
        ingestor.ingest_file(str(staged), embeddings=None, db=object())


def make_patch(path, rel="chroma.sqlite3", content=b"new index", expected=None, backend="chroma"):
    digest = hashlib.sha256(content).hexdigest() if expected is None else expected
    manifest = {
        "format": 2, "version": "test-v1", "vector_backend": backend,
        "embedding_model": "nomic-embed-text", "files": {rel: digest},
    }
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        zf.writestr(f"vectorstore/{rel}", content)
    return path


def test_patch_checksum_and_archive_path_validation(tmp_path):
    patcher = load_script("apply_data_patch")
    good = make_patch(tmp_path / "good.zip")
    assert patcher.validate_patch(good)["version"] == "test-v1"
    bad_hash = make_patch(tmp_path / "bad-hash.zip", expected="0" * 64)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        patcher.validate_patch(bad_hash)
    traversal = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal, "w") as zf:
        zf.writestr("manifest.json", "{}")
        zf.writestr("../escape", "bad")
    with pytest.raises(ValueError, match="Unsafe archive path"):
        patcher.validate_patch(traversal)


def test_patch_replacement_keeps_backup_and_replaces_atomically(tmp_path, monkeypatch):
    patcher = load_script("apply_data_patch")
    target = tmp_path / "db"
    target.mkdir()
    (target / "chroma.sqlite3").write_bytes(b"old index")
    patch = make_patch(tmp_path / "update.zip")
    monkeypatch.setattr(patcher, "VECTOR_PATH", str(target))
    monkeypatch.setattr(patcher, "BACKUPS_DIR", tmp_path / "backups")
    monkeypatch.setattr(patcher, "UPDATE_LOG", tmp_path / "update_log.json")
    monkeypatch.setattr(patcher, "_validate_database", lambda path, backend: None)
    result = patcher.apply_patch(patch)
    assert result["version"] == "test-v1"
    assert (target / "chroma.sqlite3").read_bytes() == b"new index"
    backup = Path(result["backup_path"])
    assert (backup / "chroma.sqlite3").read_bytes() == b"old index"

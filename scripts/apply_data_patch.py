#!/usr/bin/env python3
"""Verify, validate and atomically apply a signed knowledge-base ZIP patch."""
import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from vet_ai_tutor.config import (
    VECTOR_PATH, VECTOR_BACKEND, BACKUPS_DIR, EMBEDDING_MODEL,
    PATCH_PUBLIC_KEY_PATH, ALLOW_UNSIGNED_PATCHES, MAX_PATCH_BYTES, project_path,
)
from vet_ai_tutor.embeddings import embedding_profile
from vet_ai_tutor.signing import SIGNATURE_FILE, verify

UPDATE_LOG = PROJECT_ROOT / "data" / "update_log.json"


def _safe_members(zf: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, str]]:
    result = []
    total = 0
    for info in zf.infolist():
        name = info.filename
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError(f"Unsafe archive path: {name}")
        mode = info.external_attr >> 16
        if mode & 0o170000 == 0o120000:
            raise ValueError(f"Symlinks are not allowed in patches: {name}")
        total += info.file_size
        if total > MAX_PATCH_BYTES:
            raise ValueError("Patch exceeds configured uncompressed size limit")
        result.append((info, name))
    return result


def _check_signature(zf: zipfile.ZipFile, names: set, manifest_bytes: bytes,
                     public_key_path: str | Path | None, allow_unsigned: bool) -> None:
    key = Path(project_path(str(public_key_path))) if public_key_path else None
    if key is None or not key.is_file():
        if allow_unsigned:
            print("WARNING: no patch public key configured; accepting patch without signature check.")
            return
        raise ValueError(
            f"No trusted patch public key at {key}. Install the developer's "
            "patch_signing_public.pem, or set VET_ALLOW_UNSIGNED_PATCHES=1 for development."
        )
    if SIGNATURE_FILE not in names:
        raise ValueError("Patch is not signed")
    verify(manifest_bytes, zf.read(SIGNATURE_FILE), key)


def validate_patch(patch_path: str | Path, expected_backend: str = VECTOR_BACKEND,
                   expected_embedding_model: str = EMBEDDING_MODEL,
                   public_key_path: str | Path | None = None,
                   allow_unsigned: bool | None = None) -> dict:
    public_key_path = PATCH_PUBLIC_KEY_PATH if public_key_path is None else public_key_path
    allow_unsigned = ALLOW_UNSIGNED_PATCHES if allow_unsigned is None else allow_unsigned
    with zipfile.ZipFile(patch_path, "r") as zf:
        members = _safe_members(zf)
        names = {name for _, name in members}
        if "manifest.json" not in names:
            raise ValueError("Patch manifest is missing")
        manifest_bytes = zf.read("manifest.json")
        _check_signature(zf, names, manifest_bytes, public_key_path, allow_unsigned)
        manifest = json.loads(manifest_bytes)
        if manifest.get("format") != 2:
            raise ValueError("Unsupported patch format")
        if not isinstance(manifest.get("version"), str) or not manifest["version"].strip():
            raise ValueError("Patch version is missing")
        if manifest.get("vector_backend") != expected_backend:
            raise ValueError("Patch vector backend does not match this installation")
        if manifest.get("embedding_model") != expected_embedding_model:
            raise ValueError("Patch embedding model does not match this installation")
        profile = manifest.get("embedding_profile")
        expected_profile = embedding_profile(expected_embedding_model)
        if profile is not None and any(
            profile.get(k) != expected_profile[k] for k in ("query_prefix", "document_prefix")
        ):
            raise ValueError("Patch embedding prefixes do not match this installation")
        entries = manifest.get("files")
        if not isinstance(entries, dict) or not entries:
            raise ValueError("Patch manifest has no file checksums")
        for rel, digest in entries.items():
            rel_path = PurePosixPath(rel)
            if rel_path.is_absolute() or ".." in rel_path.parts or "\\" in rel:
                raise ValueError(f"Unsafe manifest path: {rel}")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"Invalid SHA-256 checksum for {rel}")
        expected_names = {f"vectorstore/{name}" for name in entries}
        if names - {"manifest.json", SIGNATURE_FILE} != expected_names:
            raise ValueError("Patch files do not match the manifest")
        for rel, expected_hash in entries.items():
            digest = hashlib.sha256(zf.read(f"vectorstore/{rel}")).hexdigest()
            if digest != expected_hash:
                raise ValueError(f"Checksum mismatch for {rel}")
    return manifest


def _validate_database(path: Path, backend: str):
    if backend == "chroma":
        valid = (path / "chroma.sqlite3").is_file()
    elif backend == "faiss":
        valid = all((path / n).is_file() for n in ("index.faiss", "index.pkl"))
    else:
        valid = False
    if not valid:
        raise ValueError("Patch does not contain a complete vector database")
    if backend == "chroma":
        import chromadb
        client = chromadb.PersistentClient(path=str(path))
        collections = client.list_collections()
        if not collections:
            raise ValueError("Patch vector database has no collections")
        required = ("source_title", "author", "journal", "publication_year", "source_url", "source_tier")
        count = 0
        for collection in collections:
            offset = 0
            while True:
                records = collection.get(include=["metadatas"], limit=1000, offset=offset)
                metadata_rows = records.get("metadatas") or []
                for metadata in metadata_rows:
                    missing = [key for key in required if not str((metadata or {}).get(key, "")).strip()]
                    if missing or metadata.get("source_tier") not in {"tier_1", "tier_2"}:
                        raise ValueError("Patch contains vectors with incomplete or unapproved provenance")
                    if not re.fullmatch(r"(?:18|19|20|21)\d{2}", str(metadata.get("publication_year", ""))):
                        raise ValueError("Patch contains vectors with invalid publication years")
                    count += 1
                if len(metadata_rows) < 1000:
                    break
                offset += len(metadata_rows)
        if count == 0:
            raise ValueError("Patch vector database contains no vectors")


def apply_patch(patch_path: str | Path, no_backup: bool = False) -> dict:
    patch_path = Path(patch_path).resolve()
    if not patch_path.is_file() or not zipfile.is_zipfile(patch_path):
        raise ValueError("Patch file is missing or is not a ZIP archive")
    manifest = validate_patch(patch_path)
    target = Path(VECTOR_PATH).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    backup_root = Path(BACKUPS_DIR)
    backup_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    backup = backup_root / f"backup_{VECTOR_BACKEND}_{stamp}"
    stage = Path(tempfile.mkdtemp(prefix=".vet-patch-", dir=target.parent))
    displaced = False
    try:
        with zipfile.ZipFile(patch_path, "r") as zf:
            for info, name in _safe_members(zf):
                if name in ("manifest.json", SIGNATURE_FILE):
                    continue
                rel = PurePosixPath(name).relative_to("vectorstore")
                out = stage.joinpath(*rel.parts)
                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, out.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        _validate_database(stage, VECTOR_BACKEND)
        if target.exists() and not no_backup:
            shutil.copytree(target, backup)
        if target.exists():
            target.rename(target.parent / f".{target.name}.previous-{stamp}")
            displaced = True
            previous = target.parent / f".{target.name}.previous-{stamp}"
        else:
            previous = None
        try:
            stage.rename(target)
        except Exception:
            if displaced and previous and previous.exists():
                previous.rename(target)
                displaced = False
            raise
        if previous and previous.exists():
            shutil.rmtree(previous)
        entry = {
            "patch_file": patch_path.name, "version": manifest["version"],
            "vector_backend": VECTOR_BACKEND, "applied_at": datetime.now(timezone.utc).isoformat(),
            "backup_path": str(backup) if backup.exists() else "",
        }
        UPDATE_LOG.parent.mkdir(parents=True, exist_ok=True)
        try:
            history = json.loads(UPDATE_LOG.read_text(encoding="utf-8")) if UPDATE_LOG.exists() else []
        except json.JSONDecodeError:
            history = []
        history.append(entry)
        UPDATE_LOG.write_text(json.dumps(history, indent=2), encoding="utf-8")
        return {**entry, "backup_path": str(backup) if backup.exists() else ""}
    except Exception:
        if displaced and not target.exists():
            previous = target.parent / f".{target.name}.previous-{stamp}"
            if previous.exists():
                previous.rename(target)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def rollback() -> Path:
    root = Path(BACKUPS_DIR)
    backups = sorted(p for p in root.glob(f"backup_{VECTOR_BACKEND}_*") if p.is_dir())
    if not backups:
        raise ValueError("No backup is available for rollback")
    backup = backups[-1]
    target = Path(VECTOR_PATH).resolve()
    _validate_database(backup, VECTOR_BACKEND)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    temporary = target.parent / f".{target.name}.rollback-{stamp}"
    shutil.copytree(backup, temporary)
    current_backup = target.parent / f".{target.name}.before-rollback-{stamp}"
    try:
        if target.exists():
            target.rename(current_backup)
        temporary.rename(target)
        if current_backup.exists():
            shutil.rmtree(current_backup)
    except Exception:
        if not target.exists() and current_backup.exists():
            current_backup.rename(target)
        raise
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch")
    parser.add_argument("--rollback", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    try:
        if args.rollback:
            print(f"Restored {rollback()}")
        elif args.patch:
            result = apply_patch(args.patch, args.no_backup)
            print(f"Applied patch {result['version']}; restart the tutor to reload the index.")
        else:
            parser.error("Provide --patch <file> or --rollback")
    except Exception as exc:
        parser.exit(1, f"Patch operation failed: {exc}\n")


if __name__ == "__main__":
    main()

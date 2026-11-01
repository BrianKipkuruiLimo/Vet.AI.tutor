#!/usr/bin/env python3
"""Create a checksummed, signed, versioned full knowledge-base patch."""
import argparse
import hashlib
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from vet_ai_tutor.config import (
    VECTOR_PATH, VECTOR_BACKEND, EMBEDDING_MODEL, LLM_MODEL, PATCH_SIGNING_KEY_PATH,
)
from vet_ai_tutor.embeddings import embedding_profile
from vet_ai_tutor.signing import SIGNATURE_FILE, sign

REQUIRED_METADATA = ("source_title", "author", "journal", "publication_year", "source_url", "source_tier")


def validate_index(source: Path | str | None = None) -> int:
    """Require a nonempty index whose every Chroma vector has complete provenance."""
    source = Path(source or VECTOR_PATH).resolve()
    if not source.is_dir() or not (source / "chroma.sqlite3").is_file():
        raise ValueError(f"Chroma database is missing or incomplete: {source}")
    import chromadb
    client = chromadb.PersistentClient(path=str(source))
    collections = client.list_collections()
    if not collections:
        raise ValueError("Chroma database has no collections")
    total = 0
    for collection in collections:
        offset = 0
        while True:
            result = collection.get(include=["metadatas"], limit=1000, offset=offset)
            metadatas = result.get("metadatas") or []
            for metadata in metadatas:
                missing = [key for key in REQUIRED_METADATA if not str((metadata or {}).get(key, "")).strip()]
                if missing:
                    raise ValueError(f"Vector in {collection.name} lacks provenance: {', '.join(missing)}")
                if metadata.get("source_tier") not in {"tier_1", "tier_2"}:
                    raise ValueError(f"Vector in {collection.name} has unapproved source tier")
                if not re.fullmatch(r"(?:18|19|20|21)\d{2}", str(metadata["publication_year"])):
                    raise ValueError(f"Vector in {collection.name} has invalid publication year")
                total += 1
            if len(metadatas) < 1000:
                break
            offset += len(metadatas)
    if total == 0:
        raise ValueError("Chroma database contains no vectors")
    return total


def create_patch(version: str | None = None, output_dir: str | None = None,
                 signing_key: str | Path | None = PATCH_SIGNING_KEY_PATH) -> Path:
    """Package the active index. Pass signing_key=None to build an unsigned patch."""
    if signing_key is not None and not Path(signing_key).is_file():
        raise ValueError(
            f"Signing key not found: {signing_key}. Create one with "
            "`python scripts/generate_signing_key.py`, or pass --unsigned for local testing."
        )
    source = Path(VECTOR_PATH).resolve()
    if VECTOR_BACKEND == "chroma":
        validate_index(source)
    elif not source.is_dir() or not any(source.iterdir()):
        raise ValueError(f"Vector database is missing or empty: {source}")
    if VECTOR_BACKEND == "faiss" and not all((source / n).is_file() for n in ("index.faiss", "index.pkl")):
        raise ValueError("FAISS database is incomplete")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    version = version or f"v{stamp}"
    files = {}
    for item in sorted(source.rglob("*")):
        if item.is_file():
            rel = item.relative_to(source).as_posix()
            files[rel] = hashlib.sha256(item.read_bytes()).hexdigest()
    manifest = {
        "format": 2, "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "vector_backend": VECTOR_BACKEND, "embedding_model": EMBEDDING_MODEL,
        "embedding_profile": embedding_profile(),
        "llm_model": LLM_MODEL, "file_count": len(files), "files": files,
    }
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
    out = Path(output_dir or PROJECT_ROOT / "data" / "patches").resolve()
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"vet_ai_data_patch_{version}.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("manifest.json", manifest_bytes)
        if signing_key is not None:
            zf.writestr(SIGNATURE_FILE, sign(manifest_bytes, signing_key))
        for rel in files:
            zf.write(source / rel, f"vectorstore/{rel}")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version")
    parser.add_argument("--output")
    parser.add_argument("--signing-key", default=PATCH_SIGNING_KEY_PATH,
                        help="Ed25519 private key (default: %(default)s)")
    parser.add_argument("--unsigned", action="store_true",
                        help="Build without a signature (only accepted where unsigned patches are allowed)")
    args = parser.parse_args()
    try:
        result = create_patch(args.version, args.output, None if args.unsigned else args.signing_key)
        print(f"Created {result} ({result.stat().st_size / 1024 / 1024:.1f} MiB)")
    except Exception as exc:
        parser.exit(1, f"Patch creation failed: {exc}\n")


if __name__ == "__main__":
    main()

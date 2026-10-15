#!/usr/bin/env python3
"""Rebuild vectorstore from RAW_DIR using Ollama embeddings (lightweight).

Supports ChromaDB (recommended) and FAISS (legacy fallback).
Embeddings are generated via Ollama — no PyTorch / sentence-transformers needed.

Usage:
    python scripts/rebuild_vectorstore.py

Prerequisites:
    1. Ollama running:   ollama serve
    2. Embedding model:  ollama pull nomic-embed-text
"""
import os
import sys
import shutil
import traceback
import json
import tempfile
from datetime import datetime, timezone
from urllib.parse import urlparse

try:
    from vet_ai_tutor.config import (
        RAW_DIR, VECTOR_PATH, EMBEDDING_MODEL, EMBEDDING_BACKEND,
        CHUNK_SIZE, CHUNK_OVERLAP, SKIP_EXTENSIONS, SUPPORTED_EXTENSIONS,
        VECTOR_BACKEND,
        SOURCE_CATALOG_PATH,
        WHITELIST_PATH,
        BACKUPS_DIR,
    )
except Exception:
    sys.path.insert(0, os.getcwd())
    from vet_ai_tutor.config import (
        RAW_DIR, VECTOR_PATH, EMBEDDING_MODEL, EMBEDDING_BACKEND,
        CHUNK_SIZE, CHUNK_OVERLAP, SKIP_EXTENSIONS, SUPPORTED_EXTENSIONS,
        VECTOR_BACKEND,
        SOURCE_CATALOG_PATH,
        WHITELIST_PATH,
        BACKUPS_DIR,
    )

print("RAW_DIR:          ", RAW_DIR)
print("VECTOR_PATH:      ", VECTOR_PATH)
print("EMBEDDING_BACKEND:", EMBEDDING_BACKEND)
print("EMBEDDING_MODEL:  ", EMBEDDING_MODEL)
print("VECTOR_BACKEND:   ", VECTOR_BACKEND)
print("CHUNK_SIZE/OVERLAP:", CHUNK_SIZE, CHUNK_OVERLAP)
print()

# ---------------------------------------------------------------------------
# Document loaders (no unstructured dependency)
# ---------------------------------------------------------------------------
try:
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception as e:
    print("Required packages missing:", e)
    raise SystemExit(1)


def _load_pptx(path: str):
    """Load a .pptx file using python-pptx (lightweight, no unstructured needed)."""
    try:
        from pptx import Presentation
        from langchain_core.documents import Document
        prs = Presentation(path)
        texts = []
        for slide_num, slide in enumerate(prs.slides, 1):
            parts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    parts.append(shape.text.strip())
            if parts:
                texts.append(
                    Document(
                        page_content="\n".join(parts),
                        metadata={"source": os.path.basename(path), "slide": slide_num},
                    )
                )
        return texts
    except Exception as e:
        print(f"  Warning: pptx load failed for {path}: {e}")
        return []


def _get_loader(path: str, ext: str):
    """Return loaded documents for the given file."""
    kind = SUPPORTED_EXTENSIONS.get(ext)
    if kind == "pdf":
        return PyPDFLoader(path).load()
    elif kind == "docx":
        return Docx2txtLoader(path).load()
    elif kind in ("pptx", "ppt"):
        return _load_pptx(path)
    elif kind == "txt":
        return TextLoader(path, encoding="utf-8").load()
    elif kind == "json":
        return TextLoader(path, encoding="utf-8").load()
    elif kind == "doc":
        # Try docx loader first (many .doc files are actually docx)
        try:
            return Docx2txtLoader(path).load()
        except Exception:
            return []
    return []


# ---------------------------------------------------------------------------
# Resolve vectorstore backend
# ---------------------------------------------------------------------------
VectorStore = None
backend = None

if VECTOR_BACKEND == "chroma":
    try:
        from langchain_chroma import Chroma
        VectorStore = Chroma
        backend = "chroma"
        print("Using ChromaDB backend.")
    except ImportError:
        print("ChromaDB not installed; falling back to FAISS.")

if VectorStore is None:
    try:
        from langchain_community.vectorstores import FAISS
        VectorStore = FAISS
        backend = "faiss"
        print("Using FAISS backend.")
    except ImportError:
        print("Neither ChromaDB nor FAISS is installed; aborting.")
        raise SystemExit(1)

# ---------------------------------------------------------------------------
# Load embeddings — Ollama first, HuggingFace fallback
# ---------------------------------------------------------------------------
embeddings = None

if EMBEDDING_BACKEND == "ollama":
    try:
        from langchain_ollama import OllamaEmbeddings
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        # Smoke test
        embeddings.embed_query("test")
        print(f"Using OllamaEmbeddings: {EMBEDDING_MODEL}")
    except Exception as e:
        print(f"OllamaEmbeddings failed: {e}")
        print("Make sure Ollama is running and you have pulled the model:")
        print(f"  ollama pull {EMBEDDING_MODEL}")
        embeddings = None

if embeddings is None:
    print("Falling back to HuggingFace embeddings (requires sentence-transformers)...")
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        hf_model = EMBEDDING_MODEL if EMBEDDING_BACKEND == "huggingface" else "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=hf_model)
        print(f"Using HuggingFaceEmbeddings: {hf_model}")
    except Exception as e:
        print(f"HuggingFace embeddings also failed: {e}")
        raise SystemExit(1)

# ---------------------------------------------------------------------------
# Gather and load documents
# ---------------------------------------------------------------------------
documents = []
failed = []
skipped = []
catalog_path = SOURCE_CATALOG_PATH if os.path.isabs(SOURCE_CATALOG_PATH) else os.path.join(os.getcwd(), SOURCE_CATALOG_PATH)
if not os.path.isfile(catalog_path):
    raise SystemExit(f"Required provenance catalog missing: {catalog_path}. Create it and review every source before rebuilding.")
with open(catalog_path, "r", encoding="utf-8") as catalog_file:
    source_catalog = json.load(catalog_file)
whitelist_path = WHITELIST_PATH if os.path.isabs(WHITELIST_PATH) else os.path.join(os.getcwd(), WHITELIST_PATH)
with open(whitelist_path, "r", encoding="utf-8") as whitelist_file:
    whitelist_data = json.load(whitelist_file)
approved_domains = {
    domain.lower().strip("."): tier
    for tier, group in whitelist_data.items() if tier in {"tier_1", "tier_2"}
    for domain in group.get("domains", [])
}
if not isinstance(source_catalog, dict):
    raise SystemExit("Source metadata catalog must be a JSON object keyed by source filename.")
required_provenance = ("source_title", "author", "journal", "publication_year", "source_url", "source_tier")

for root, _, files in os.walk(RAW_DIR):
    for fn in sorted(files):
        path = os.path.join(root, fn)
        ext = os.path.splitext(fn.lower())[1]

        if ext in SKIP_EXTENSIONS or not ext:
            skipped.append(path)
            continue

        if ext not in SUPPORTED_EXTENSIONS:
            skipped.append(path)
            continue

        try:
            loaded = _get_loader(path, ext)
            if not loaded:
                skipped.append(path)
                continue

            # Enrich metadata
            source_name = os.path.basename(path)
            source_key = os.path.relpath(path, RAW_DIR).replace(os.sep, "/")
            provenance = source_catalog.get(source_key) or source_catalog.get(source_name)
            if not isinstance(provenance, dict):
                raise ValueError(f"No reviewed provenance entry for {source_name}")
            missing = [key for key in required_provenance if not str(provenance.get(key, "")).strip()]
            if missing:
                raise ValueError(f"Missing provenance fields for {source_name}: {', '.join(missing)}")
            if str(provenance["source_tier"]) not in {"tier_1", "tier_2"}:
                raise ValueError(f"Unapproved source tier for {source_name}")
            if not str(provenance["publication_year"]).isdigit() or len(str(provenance["publication_year"])) != 4:
                raise ValueError(f"Invalid publication_year for {source_name}")
            parsed_source = urlparse(str(provenance["source_url"]))
            if parsed_source.scheme not in {"https", "file"}:
                raise ValueError(f"source_url must use https or file:// for {source_name}")
            if parsed_source.scheme == "https":
                host_parts = (parsed_source.hostname or "").lower().split(".")
                matched_tier = next((approved_domains.get(".".join(host_parts[i:])) for i in range(len(host_parts))
                                     if ".".join(host_parts[i:]) in approved_domains), None)
                if matched_tier != provenance["source_tier"]:
                    raise ValueError(f"Source URL is not whitelisted at the declared tier: {source_name}")
            valid = []
            for d in loaded:
                md = getattr(d, "metadata", {}) or {}
                md["source"] = source_name
                md["source_path"] = path
                md.update({key: str(provenance[key]) for key in required_provenance})
                md["source_domain"] = parsed_source.hostname or "local_document"
                d.metadata = md
                content = (getattr(d, "page_content", "") or "").strip()
                if len(content) >= 20:
                    valid.append(d)

            documents.extend(valid)
            print(f"Loaded {fn}: {len(valid)} pages/items")
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            failed.append({"path": path, "error": str(e)})

print(f"\n✅ Loaded {len(documents)} document pages/items")

# ---------------------------------------------------------------------------
# Split into chunks
# ---------------------------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(documents)
for i, c in enumerate(chunks):
    md = getattr(c, "metadata", {}) or {}
    md.setdefault("source", "unknown")
    md["chunk"] = i
    c.metadata = md

print(f"✅ Created {len(chunks)} text chunks")
if not chunks:
    raise SystemExit("No reviewed source chunks are ready; refusing to replace the current vector database.")

# ---------------------------------------------------------------------------
# Build and save vectorstore
# ---------------------------------------------------------------------------
try:
    vector_target = os.path.abspath(VECTOR_PATH)
    target_parent = os.path.dirname(vector_target)
    os.makedirs(target_parent, exist_ok=True)
    build_path = tempfile.mkdtemp(prefix=".vet-index-rebuild-", dir=target_parent)

    print(f"Building {backend} index (this may take a while)...")

    if backend == "chroma":
        # Batch insert to avoid memory spikes on large collections
        BATCH = 500
        db = None
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i:i + BATCH]
            pct = min(i + BATCH, len(chunks))
            print(f"  Embedding batch {i}–{pct} / {len(chunks)}...")
            if db is None:
                db = VectorStore.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory=build_path,
                )
            else:
                db.add_documents(batch)
        doc_count = db._collection.count() if db else 0
        print(f"✅ ChromaDB built in staging directory ({doc_count} documents)")
    else:  # faiss
        db = VectorStore.from_documents(chunks, embeddings)
        db.save_local(build_path)
        print(f"✅ FAISS index built at {VECTOR_PATH} ({len(chunks)} chunks)")

    backup_path = None
    previous_path = vector_target + ".previous"
    if os.path.exists(vector_target):
        backup_root = BACKUPS_DIR if os.path.isabs(BACKUPS_DIR) else os.path.join(os.getcwd(), BACKUPS_DIR)
        os.makedirs(backup_root, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        backup_path = os.path.join(backup_root, f"backup_{backend}_rebuild_{stamp}")
        shutil.copytree(vector_target, backup_path)
        if os.path.exists(previous_path):
            shutil.rmtree(previous_path)
        os.replace(vector_target, previous_path)
    try:
        os.replace(build_path, vector_target)
    except Exception:
        if os.path.exists(previous_path) and not os.path.exists(vector_target):
            os.replace(previous_path, vector_target)
        raise
    if os.path.exists(previous_path):
        shutil.rmtree(previous_path)
    print(f"✅ New index installed at {vector_target}")

except Exception:
    if 'build_path' in locals() and os.path.exists(build_path):
        shutil.rmtree(build_path, ignore_errors=True)
    print("Failed building vectorstore:")
    traceback.print_exc()
    raise

# ---------------------------------------------------------------------------
# Write summary
# ---------------------------------------------------------------------------
try:
    import json
    out = os.path.join("archive", "rebuild_summary.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "loaded_pages": len(documents),
            "chunks": len(chunks),
            "failed": failed,
            "skipped": len(skipped),
            "vector_path": VECTOR_PATH,
            "embedding_backend": EMBEDDING_BACKEND,
            "embedding_model": EMBEDDING_MODEL,
            "vector_backend": backend,
        }, f, indent=2)
    print("Summary written:", out)
except Exception as e:
    print("Could not write summary:", e)

print("\nDone.")

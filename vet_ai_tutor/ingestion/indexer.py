"""Chunking and vector-store operations shared by the rebuild and staging paths."""
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

from vet_ai_tutor.config import (
    BACKUPS_DIR, CHUNK_OVERLAP, CHUNK_SIZE, VECTOR_BACKEND, VECTOR_PATH, project_path,
)
from vet_ai_tutor.embeddings import check_index_profile, read_index_info, write_index_info

logger = logging.getLogger(__name__)

EMBED_BATCH = 100
# Files inside the index directory that a rebuild carries over to the new index.
PRESERVED_FILES = ("quiz_bank.json",)


def vectorstore_class() -> Tuple[Any, str]:
    """Return (VectorStore class, backend name): Chroma, else FAISS."""
    if VECTOR_BACKEND == "chroma":
        try:
            from langchain_chroma import Chroma
            return Chroma, "chroma"
        except ImportError:
            logger.info("ChromaDB not installed; falling back to FAISS.")
    try:
        from langchain_community.vectorstores import FAISS
        return FAISS, "faiss"
    except ImportError as exc:
        raise RuntimeError("Neither ChromaDB nor FAISS is installed") from exc


def chunk_documents(documents: List, chunk_size: int = CHUNK_SIZE,
                    chunk_overlap: int = CHUNK_OVERLAP) -> List:
    """Split page/slide documents into chunks numbered in order."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata.setdefault("source", "unknown")
        chunk.metadata["chunk"] = i
    return chunks


def open_vectorstore(embeddings, path: str = VECTOR_PATH):
    """Open (or lazily create) the active index. FAISS returns None until first add."""
    VectorStore, backend = vectorstore_class()
    if backend == "chroma":
        db = VectorStore(persist_directory=path, embedding_function=embeddings)
        logger.info("Opened ChromaDB at %s (%d existing docs)", path, db._collection.count())
        return db
    if os.path.exists(os.path.join(path, "index.faiss")):
        return VectorStore.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return None


def add_documents(db, documents: List, embeddings, path: str = VECTOR_PATH):
    """Add documents to the active index in batches; returns the (possibly new) db."""
    if not documents:
        return db
    VectorStore, backend = vectorstore_class()
    if backend == "chroma":
        for i in range(0, len(documents), EMBED_BATCH):
            logger.info("  Embedding batch %d–%d / %d...", i, min(i + EMBED_BATCH, len(documents)), len(documents))
            db.add_documents(documents[i:i + EMBED_BATCH])
        logger.info("Added %d documents to ChromaDB (total: %d)", len(documents), db._collection.count())
    else:
        new_db = VectorStore.from_documents(documents, embeddings)
        if db is None:
            db = new_db
        else:
            db.merge_from(new_db)
        os.makedirs(path, exist_ok=True)
        db.save_local(path)
        logger.info("Added %d documents to FAISS.", len(documents))
    _record_index_profile(path)
    return db


def _record_index_profile(path: str) -> None:
    """Write index_info.json on first add; warn if the index uses another profile."""
    if read_index_info(path) is None:
        write_index_info(path)
        return
    warning = check_index_profile(path)
    if warning:
        logger.warning(warning)


def build_index(chunks: List, embeddings, target: str = VECTOR_PATH,
                backups_dir: str = BACKUPS_DIR, progress=print) -> Optional[str]:
    """Build a new index beside `target`, back up the old one, and swap atomically.

    Returns the backup path (None if there was no previous index). On any
    failure the previous index stays in place.
    """
    if not chunks:
        raise ValueError("No chunks to index; refusing to replace the current vector database.")
    VectorStore, backend = vectorstore_class()
    target = os.path.abspath(project_path(target))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    build_path = tempfile.mkdtemp(prefix=".vet-index-rebuild-", dir=os.path.dirname(target))
    try:
        if backend == "chroma":
            db = None
            started = time.time()
            for i in range(0, len(chunks), EMBED_BATCH):
                batch = chunks[i:i + EMBED_BATCH]
                if db is None:
                    db = VectorStore.from_documents(documents=batch, embedding=embeddings,
                                                    persist_directory=build_path)
                else:
                    db.add_documents(batch)
                done = i + len(batch)
                rate = done / max(time.time() - started, 1e-6)
                progress(f"  Indexed {done}/{len(chunks)} chunks "
                         f"(~{(len(chunks) - done) / rate / 60:.0f} min left)")
        else:
            VectorStore.from_documents(chunks, embeddings).save_local(build_path)
        write_index_info(build_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        for name in PRESERVED_FILES:
            if os.path.isfile(os.path.join(target, name)):
                shutil.copy2(os.path.join(target, name), os.path.join(build_path, name))

        backup_path = None
        previous_path = target + ".previous"
        if os.path.exists(target):
            backup_root = project_path(backups_dir)
            os.makedirs(backup_root, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            backup_path = os.path.join(backup_root, f"backup_{backend}_rebuild_{stamp}")
            shutil.copytree(target, backup_path)
            if os.path.exists(previous_path):
                shutil.rmtree(previous_path)
            os.replace(target, previous_path)
        try:
            os.replace(build_path, target)
        except Exception:
            if os.path.exists(previous_path) and not os.path.exists(target):
                os.replace(previous_path, target)
            raise
        if os.path.exists(previous_path):
            shutil.rmtree(previous_path)
        return backup_path
    finally:
        if os.path.exists(build_path):
            shutil.rmtree(build_path, ignore_errors=True)

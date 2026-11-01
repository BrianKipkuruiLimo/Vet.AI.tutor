"""Persistent embedding cache for index rebuilds.

Embedding ~30k chunks takes hours on a laptop CPU, and the catalog is
approved in batches, so every rebuild would otherwise re-embed everything.
Vectors are cached by SHA-256 of the chunk text in a SQLite file per
embedding profile (model + prefixes), so a rebuild only embeds new or changed
chunks and an interrupted rebuild resumes where it stopped.
"""
import hashlib
import json
import os
import sqlite3
import threading
from array import array
from typing import List, Optional

from langchain_core.embeddings import Embeddings

from vet_ai_tutor.config import project_path
from vet_ai_tutor.embeddings import embedding_profile


def default_cache_path(profile: Optional[dict] = None) -> str:
    from vet_ai_tutor.ingestion.loaders import CACHE_DIR
    key = hashlib.sha256(json.dumps(profile or embedding_profile(), sort_keys=True).encode()).hexdigest()
    return os.path.join(project_path(CACHE_DIR), "embeddings", f"{key[:16]}.sqlite")


class CachedEmbeddings(Embeddings):
    """Wrap an Embeddings object; document vectors are read from/written to SQLite."""

    def __init__(self, base: Embeddings, path: Optional[str] = None):
        self._base = base
        self.path = path or default_cache_path()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS vectors (hash TEXT PRIMARY KEY, vector BLOB NOT NULL)")
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        hashes = [self._hash(t) for t in texts]
        with self._lock:
            found = {}
            for i in range(0, len(hashes), 500):  # stay under SQLite's variable limit
                batch = hashes[i:i + 500]
                rows = self._db.execute(
                    f"SELECT hash, vector FROM vectors WHERE hash IN ({','.join('?' * len(batch))})", batch)
                found.update({h: array("f", blob).tolist() for h, blob in rows})
        missing = [i for i, h in enumerate(hashes) if h not in found]
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        if missing:
            vectors = self._base.embed_documents([texts[i] for i in missing])
            with self._lock:
                self._db.executemany(
                    "INSERT OR REPLACE INTO vectors (hash, vector) VALUES (?, ?)",
                    [(hashes[i], array("f", v).tobytes()) for i, v in zip(missing, vectors, strict=True)],
                )
                self._db.commit()
            for i, vector in zip(missing, vectors, strict=True):
                found[hashes[i]] = list(vector)
        return [found[h] for h in hashes]

    def embed_query(self, text: str) -> List[float]:
        return self._base.embed_query(text)

    def close(self) -> None:
        with self._lock:
            self._db.close()

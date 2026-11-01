"""Shared embedding loader used by the app, ingestion scripts and evaluators.

Indexing and querying must embed text the same way, so every component loads
embeddings through `load_embeddings()` and records the result with
`write_index_info()` next to the index it builds.
"""
import json
import logging
import os
from typing import Any, List, Optional, Tuple

from langchain_core.embeddings import Embeddings

from vet_ai_tutor.config import (
    EMBEDDING_BACKEND, EMBEDDING_MODEL, EMBEDDING_PREFIXES, OFFLINE_ONLY,
)

logger = logging.getLogger(__name__)

INDEX_INFO_FILE = "index_info.json"

# (query prefix, document prefix) the model was trained with, matched by name
# prefix. Taken from each model's published usage instructions.
_KNOWN_PREFIXES = {
    "nomic-embed-text": ("search_query: ", "search_document: "),
    "mxbai-embed-large": ("Represent this sentence for searching relevant passages: ", ""),
    "embeddinggemma": ("task: search result | query: ", "title: none | text: "),
    "qwen3-embedding": (
        "Instruct: Given a veterinary question, retrieve passages that answer it\nQuery: ", ""
    ),
}


def prefixes_for(model: str, mode: str = EMBEDDING_PREFIXES) -> Tuple[str, str]:
    """Return the (query, document) prefixes to apply for `model`."""
    if mode == "off":
        return "", ""
    name = model.rsplit("/", 1)[-1].lower()
    for known, pair in _KNOWN_PREFIXES.items():
        if name.startswith(known):
            return pair
    return "", ""


def embedding_profile(model: str = EMBEDDING_MODEL, mode: str = EMBEDDING_PREFIXES) -> dict:
    """Describe how vectors are produced, so mismatched indexes can be detected."""
    query_prefix, document_prefix = prefixes_for(model, mode)
    return {
        "embedding_backend": EMBEDDING_BACKEND,
        "embedding_model": model,
        "query_prefix": query_prefix,
        "document_prefix": document_prefix,
    }


class PrefixedEmbeddings(Embeddings):
    """Wrap an embedding model and add task prefixes to queries and documents."""

    def __init__(self, base: Embeddings, query_prefix: str, document_prefix: str):
        self._base = base
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._base.embed_documents([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> List[float]:
        return self._base.embed_query(self.query_prefix + text)


def _with_prefixes(base: Embeddings, model: str) -> Embeddings:
    query_prefix, document_prefix = prefixes_for(model)
    if not (query_prefix or document_prefix):
        return base
    logger.info("Applying embedding task prefixes for %s", model)
    return PrefixedEmbeddings(base, query_prefix, document_prefix)


def load_embeddings(
    embedding_model: str = EMBEDDING_MODEL, embedding_backend: str = EMBEDDING_BACKEND
) -> Optional[Embeddings]:
    """Load embeddings — Ollama first (no PyTorch), HuggingFace as a fallback.

    In offline mode the HuggingFace fallback only uses locally cached models,
    so it never reaches the network.
    """
    if embedding_backend == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
            emb = OllamaEmbeddings(model=embedding_model)
            # Quick smoke-test — if Ollama is not running this raises immediately
            emb.embed_query("test")
            logger.info("OllamaEmbeddings loaded: model=%s", embedding_model)
            return _with_prefixes(emb, embedding_model)
        except Exception as e:
            logger.warning(
                "OllamaEmbeddings failed (is Ollama running? did you run "
                "'ollama pull %s'?): %s", embedding_model, e,
            )
        if OFFLINE_ONLY:
            logger.error("Ollama embeddings unavailable and offline mode is enabled; refusing network fallback.")
            return None

    HFEmbeddings: Any = None
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        HFEmbeddings = HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings as CommunityHFEmbeddings
            HFEmbeddings = CommunityHFEmbeddings
        except ImportError:
            pass
    if HFEmbeddings is None:
        logger.error("No embedding backend available. Install sentence-transformers or ensure Ollama is running.")
        return None

    # Use a lightweight model if falling back from Ollama
    hf_model = embedding_model if embedding_backend == "huggingface" else "sentence-transformers/all-MiniLM-L6-v2"
    try:
        emb = HFEmbeddings(model_name=hf_model, model_kwargs={"local_files_only": True})
        logger.info("HuggingFaceEmbeddings loaded from cache: %s", hf_model)
        return _with_prefixes(emb, hf_model)
    except Exception:
        if OFFLINE_ONLY:
            logger.error("Offline mode: Hugging Face embedding model is not cached locally.")
            return None
    try:
        emb = HFEmbeddings(model_name=hf_model)
        logger.info("HuggingFaceEmbeddings downloaded: %s", hf_model)
        return _with_prefixes(emb, hf_model)
    except Exception as e:
        logger.error("HuggingFaceEmbeddings failed: %s", e)
        return None


def write_index_info(index_dir: str, profile: Optional[dict] = None, **extra) -> None:
    """Record the embedding profile an index was built with."""
    info = {**(profile or embedding_profile()), **extra}
    os.makedirs(index_dir, exist_ok=True)
    with open(os.path.join(index_dir, INDEX_INFO_FILE), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def read_index_info(index_dir: str) -> Optional[dict]:
    try:
        with open(os.path.join(index_dir, INDEX_INFO_FILE), encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def check_index_profile(index_dir: str, profile: Optional[dict] = None) -> Optional[str]:
    """Return a warning if the index was built with a different embedding profile."""
    info = read_index_info(index_dir)
    expected = profile or embedding_profile()
    if info is None:
        return (
            f"{index_dir} has no {INDEX_INFO_FILE}; it may predate embedding prefixes. "
            "Rebuild it with scripts/rebuild_vectorstore.py for best retrieval quality."
        )
    keys = ("embedding_model", "query_prefix", "document_prefix")
    if any(info.get(k) != expected.get(k) for k in keys):
        return (
            f"{index_dir} was built with {info.get('embedding_model')} "
            f"(prefixes {info.get('document_prefix')!r}) but the app is configured for "
            f"{expected['embedding_model']} (prefixes {expected['document_prefix']!r}). Rebuild the index."
        )
    return None

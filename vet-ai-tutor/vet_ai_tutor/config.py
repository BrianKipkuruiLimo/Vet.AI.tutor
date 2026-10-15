"""Central configuration for Vet AI Tutor.

All settings can be overridden via environment variables.
Values are validated at import time to catch misconfiguration early.
"""
import os
import logging

# Prevent Chroma's optional anonymous usage telemetry in offline deployments.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vector database
# ---------------------------------------------------------------------------
# Backend: "chroma" (recommended) or "faiss" (legacy fallback)
VECTOR_BACKEND = os.environ.get("VET_VECTOR_BACKEND", "chroma")
VECTOR_PATH = os.environ.get("VET_VECTOR_PATH", "data/chroma_db")

# ---------------------------------------------------------------------------
# Embedding settings
# ---------------------------------------------------------------------------
# EMBEDDING_BACKEND: "ollama" (recommended, zero extra deps) or "huggingface"
EMBEDDING_BACKEND = os.environ.get("VET_EMBEDDING_BACKEND", "ollama")
# Ollama: nomic-embed-text (fast, small). HuggingFace: all-MiniLM-L6-v2
EMBEDDING_MODEL = os.environ.get("VET_EMBEDDING_MODEL", "nomic-embed-text")

# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------
LLM_MODEL = os.environ.get("VET_LLM_MODEL", "qwen2.5:0.5b")
OLLAMA_KEEP_ALIVE = os.environ.get("VET_OLLAMA_KEEP_ALIVE", "10m")

# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------
RETRIEVER_K = int(os.environ.get("VET_RETRIEVER_K", "5"))

# ---------------------------------------------------------------------------
# Input data
# ---------------------------------------------------------------------------
RAW_DIR = os.environ.get("VET_RAW_DIR", "data/raw_notes")

# Chunking defaults
CHUNK_SIZE = int(os.environ.get("VET_CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.environ.get("VET_CHUNK_OVERLAP", "100"))

# File extensions to skip during ingestion
SKIP_EXTENSIONS = {
    ".bak", ".jpg", ".jpeg", ".png", ".gif", ".bmp",
    ".zip", ".rar", ".mp4", ".mp3", ".wav",
}

# ---------------------------------------------------------------------------
# Ingestion pipeline (developer-only)
# ---------------------------------------------------------------------------
STAGING_DIR = os.environ.get("VET_STAGING_DIR", "data/staging")
WHITELIST_PATH = os.environ.get("VET_WHITELIST_PATH", "data/whitelist.json")
SOURCE_CATALOG_PATH = os.environ.get("VET_SOURCE_CATALOG_PATH", "data/source_metadata.json")
PATCHES_DIR = os.environ.get("VET_PATCHES_DIR", "data/patches")
BACKUPS_DIR = os.environ.get("VET_BACKUPS_DIR", "data/backups")
ADMIN_TOKEN = os.environ.get("VET_ADMIN_TOKEN", "")
OFFLINE_ONLY = os.environ.get("VET_OFFLINE_ONLY", "1").lower() not in {"0", "false", "no"}

# Source credibility tiers
SOURCE_TIERS = {"tier_1", "tier_2", "tier_3", "unverified"}

# Supported document extensions and their loader types
SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "doc",
    ".pptx": "pptx",
    ".pptm": "pptx",
    ".ppt": "ppt",
    ".txt": "txt",
    ".json": "json",
    ".csv": "csv",
    ".md": "markdown",  
    
}

# ---------------------------------------------------------------------------
# History & caching
# ---------------------------------------------------------------------------
MAX_HISTORY = int(os.environ.get("VET_MAX_HISTORY", "10"))

# ---------------------------------------------------------------------------
# Production API settings
# ---------------------------------------------------------------------------
API_HOST = os.environ.get("VET_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("VET_API_PORT", "8000"))
# Comma-separated allowed origins (e.g. "http://localhost:3000,https://myapp.com")
API_CORS_ORIGINS = os.environ.get("VET_CORS_ORIGINS", "*")

# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------
_VALID_BACKENDS = {"chroma", "faiss"}
_VALID_EMBED_BACKENDS = {"ollama", "huggingface"}

if VECTOR_BACKEND not in _VALID_BACKENDS:
    raise ValueError(f"VET_VECTOR_BACKEND must be one of {_VALID_BACKENDS}, got '{VECTOR_BACKEND}'")

if EMBEDDING_BACKEND not in _VALID_EMBED_BACKENDS:
    raise ValueError(f"VET_EMBEDDING_BACKEND must be one of {_VALID_EMBED_BACKENDS}, got '{EMBEDDING_BACKEND}'")

if RETRIEVER_K < 1 or RETRIEVER_K > 50:
    raise ValueError(f"VET_RETRIEVER_K must be 1-50, got {RETRIEVER_K}")

if CHUNK_SIZE < 100:
    raise ValueError(f"VET_CHUNK_SIZE must be >= 100, got {CHUNK_SIZE}")

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError(f"VET_CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be < VET_CHUNK_SIZE ({CHUNK_SIZE})")

logger.debug(
    "Config loaded: backend=%s, embed=%s/%s, llm=%s, k=%d",
    VECTOR_BACKEND, EMBEDDING_BACKEND, EMBEDDING_MODEL, LLM_MODEL, RETRIEVER_K,
)

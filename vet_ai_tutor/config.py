"""Central configuration for Vet AI Tutor.

Every setting is a typed field on `Settings`, read from `VET_*` environment
variables or a `.env` file in the project root (environment wins). Values are
validated at import, so a bad value fails fast with the variable's name.

The module-level constants below (LLM_MODEL, VECTOR_PATH, ...) are kept for
existing imports; new code can use `get_settings()`.

    python -m vet_ai_tutor.config                 # show settings and their source
    python -m vet_ai_tutor.config --env-example   # print a documented .env template
"""
import difflib
import logging
import os
import sys
from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Prevent Chroma's optional anonymous usage telemetry in offline deployments.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")
ENV_PREFIX = "VET_"


def project_path(path: str) -> str:
    """Resolve a configured relative path against the project root."""
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def _default_signing_key() -> str:
    return os.path.join(os.path.expanduser("~"), ".config", "vet-ai-tutor", "patch_signing_key.pem")


class Settings(BaseSettings):
    """All runtime settings. Field `foo_bar` is read from `VET_FOO_BAR`."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX, env_file=ENV_FILE, env_file_encoding="utf-8",
        extra="ignore",  # unknown VET_* names are reported by warn_unknown_variables()
        validate_default=True,
    )

    # --- Vector database --------------------------------------------------
    vector_backend: Literal["chroma", "faiss"] = Field(
        "chroma", description="Vector store: chroma (recommended) or faiss (legacy fallback)")
    vector_path: str = Field("data/chroma_db", description="Directory of the active knowledge index")

    # --- Embeddings ---------------------------------------------------------
    embedding_backend: Literal["ollama", "huggingface"] = Field(
        "ollama", description="Embedding provider: ollama (no PyTorch) or huggingface")
    embedding_model: str = Field(
        "nomic-embed-text", description="Embedding model name; changing it requires an index rebuild")
    embedding_prefixes: Literal["auto", "off"] = Field(
        "auto", description="Apply the task prefixes the embedding model expects (rebuild after changing)")

    # --- LLM (defaults sized for a 2-core laptop CPU with no GPU) ------------
    llm_model: str = Field("qwen3:1.7b", description="Ollama model that writes answers")
    ollama_keep_alive: str = Field("30m", description="How long Ollama keeps the model in memory")
    llm_num_ctx: int = Field(2048, ge=512, description="Context window in tokens (prompt + passages + answer)")
    llm_num_predict: int = Field(256, ge=16, description="Maximum answer length in tokens")
    llm_temperature: float = Field(0.3, ge=0.0, le=2.0, description="Sampling temperature; low keeps answers close to sources")
    llm_timeout: int = Field(300, ge=1, description="Seconds to wait for Ollama before giving up")
    llm_reasoning: bool = Field(False, description="Qwen3 thinking mode; very slow on CPU")

    # --- Retrieval ------------------------------------------------------------
    retriever_k: int = Field(5, ge=1, le=50, description="Passages sent to the model per question")
    retrieval_candidates: int = Field(20, ge=1, description="Candidates per search (vector, BM25) before fusion")
    rrf_k: int = Field(60, ge=1, description="Reciprocal Rank Fusion constant")
    use_reranker: bool = Field(False, description="Cross-encoder re-ranking (needs sentence-transformers and a cached model)")
    cache_max: int = Field(128, ge=0, description="Cached answers kept in memory")
    cache_ttl: int = Field(3600, ge=0, description="Seconds a cached answer stays valid")

    # --- Ingestion (developer machine) ------------------------------------------
    raw_dir: str = Field("data/raw_notes", description="Source documents for full rebuilds")
    chunk_size: int = Field(600, ge=100, description="Characters per chunk; rebuild after changing")
    chunk_overlap: int = Field(100, ge=0, description="Characters shared between neighbouring chunks")
    staging_dir: str = Field("data/staging", description="Harvested and staged JSON records")
    whitelist_path: str = Field("data/whitelist.json", description="Approved web domains by tier")
    source_catalog_path: str = Field("data/source_metadata.json", description="Reviewed provenance for raw notes")
    ingest_cache_dir: str = Field("data/cache", description="OCR text, converted legacy files and embedding cache")
    ocr_min_confidence: float = Field(70.0, ge=0, le=100, description="Minimum Tesseract confidence for an OCR'd page")
    offline_only: bool = Field(True, description="Never let the embedding fallback reach the network")

    # --- Knowledge patches ------------------------------------------------------
    patches_dir: str = Field("data/patches", description="Where created patches are written")
    backups_dir: str = Field("data/backups", description="Index backups made before replacing it")
    admin_token: str = Field("", description="Enables developer routes (send as X-Admin-Token); keep secret")
    patch_public_key_path: str = Field(
        "data/patch_signing_public.pem", description="Ed25519 public key that patches must verify against")
    patch_signing_key: str = Field(
        default_factory=_default_signing_key, description="Ed25519 private key used to sign patches")
    allow_unsigned_patches: bool = Field(False, description="Accept unsigned patches when no key is installed (dev only)")
    max_patch_bytes: int = Field(20 * 1024**3, ge=1, description="Largest uncompressed patch accepted")
    max_patch_upload_bytes: int = Field(8 * 1024**3, ge=1, description="Largest patch upload through the API")

    # --- Sessions -----------------------------------------------------------------
    max_history: int = Field(10, ge=0, description="Messages remembered per conversation")
    max_sessions: int = Field(200, ge=1, description="Conversations kept in memory")
    session_ttl: int = Field(7200, ge=60, description="Seconds before an idle conversation is forgotten")

    # --- API ----------------------------------------------------------------------
    api_host: str = Field("127.0.0.1", description="Bind address; 0.0.0.0 serves a classroom LAN")
    api_port: int = Field(8000, ge=1, le=65535, description="HTTP port")
    api_reload: bool = Field(False, description="Auto-reload on code changes (development)")
    cors_origins: Optional[str] = Field(
        None, description="Comma-separated allowed origins (default: localhost on the API port)")
    rate_limit_per_minute: int = Field(20, ge=0, description="Questions per client per minute (0 disables)")
    max_question_chars: int = Field(1000, ge=10, description="Longest question accepted")

    # --- Logging ------------------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field("INFO", description="Log verbosity")
    log_format: Literal["text", "json"] = Field("text", description="text for people, json for tools")

    @field_validator("embedding_prefixes", "log_format", mode="before")
    @classmethod
    def _lower(cls, value):
        return value.lower() if isinstance(value, str) else value

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper(cls, value):
        return value.upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check_relations(self):
        if self.retrieval_candidates < self.retriever_k:
            raise ValueError(f"VET_RETRIEVAL_CANDIDATES ({self.retrieval_candidates}) must be "
                             f">= VET_RETRIEVER_K ({self.retriever_k})")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"VET_CHUNK_OVERLAP ({self.chunk_overlap}) must be < "
                             f"VET_CHUNK_SIZE ({self.chunk_size})")
        if self.cors_origins is None:
            self.cors_origins = f"http://localhost:{self.api_port},http://127.0.0.1:{self.api_port}"
        return self


def env_name(field: str) -> str:
    return ENV_PREFIX + field.upper()


def _dotenv_names(env_file: Optional[str]) -> dict:
    if not env_file or not os.path.isfile(env_file):
        return {}
    from dotenv import dotenv_values
    return {k: v for k, v in dotenv_values(env_file).items() if k}


def unknown_variables(environ=None, env_file: Optional[str] = ENV_FILE) -> dict:
    """Map unknown VET_* names (environment or .env) to a suggested correct name."""
    known = {env_name(f) for f in Settings.model_fields}
    names = [n for n in list(environ if environ is not None else os.environ) + list(_dotenv_names(env_file))
             if n.startswith(ENV_PREFIX)]
    return {n: (difflib.get_close_matches(n, known, n=1) or [None])[0] for n in names if n not in known}


def warn_unknown_variables(environ=None, env_file: Optional[str] = ENV_FILE) -> None:
    for name, suggestion in unknown_variables(environ, env_file).items():
        hint = f" — did you mean {suggestion}?" if suggestion else ""
        logger.warning("Unknown setting %s is ignored%s", name, hint)


def load_settings(env_file: Optional[str] = ENV_FILE, **overrides) -> Settings:
    """Build Settings, turning validation errors into messages that name VET_* variables."""
    try:
        return Settings(_env_file=env_file, **overrides)  # type: ignore[call-arg]
    except ValidationError as exc:
        lines = []
        for err in exc.errors():
            field = str(err["loc"][0]) if err["loc"] else ""
            name = env_name(field) if field else "settings"
            lines.append(f"{name}: {err['msg'].removeprefix('Value error, ')}")
        raise ValueError("Invalid configuration:\n  " + "\n  ".join(lines)) from None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def describe(settings: Settings, environ=None, env_file: Optional[str] = ENV_FILE) -> list:
    """Rows of (variable, value, source) with secrets masked."""
    environ = os.environ if environ is None else environ
    dotenv = _dotenv_names(env_file)
    rows = []
    for field in Settings.model_fields:
        name = env_name(field)
        value = getattr(settings, field)
        if field == "admin_token":
            value = "(set)" if value else "(not set)"
        source = "environment" if name in environ else ".env" if name in dotenv else "default"
        rows.append((name, str(value), source))
    return rows


def env_example() -> str:
    """A documented .env template with every setting commented out at its default."""
    lines = ["# Vet AI Tutor settings. Copy to .env and uncomment what you change.",
             "# Environment variables override this file. Generated by:",
             "#   python -m vet_ai_tutor.config --env-example > .env.example", ""]
    for field, info in Settings.model_fields.items():
        default = "" if field in {"patch_signing_key", "cors_origins", "admin_token"} else info.get_default(
            call_default_factory=False)
        if isinstance(default, bool):
            default = "1" if default else "0"
        lines.append(f"# {info.description}")
        lines.append(f"# {env_name(field)}={default}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level constants (existing imports use these)
# ---------------------------------------------------------------------------
settings = get_settings()

VECTOR_BACKEND = settings.vector_backend
VECTOR_PATH = settings.vector_path
EMBEDDING_BACKEND = settings.embedding_backend
EMBEDDING_MODEL = settings.embedding_model
EMBEDDING_PREFIXES = settings.embedding_prefixes
LLM_MODEL = settings.llm_model
OLLAMA_KEEP_ALIVE = settings.ollama_keep_alive
LLM_NUM_CTX = settings.llm_num_ctx
LLM_NUM_PREDICT = settings.llm_num_predict
LLM_TEMPERATURE = settings.llm_temperature
LLM_TIMEOUT = settings.llm_timeout
LLM_REASONING = settings.llm_reasoning
RETRIEVER_K = settings.retriever_k
RETRIEVAL_CANDIDATES = settings.retrieval_candidates
RRF_K = settings.rrf_k
USE_RERANKER = settings.use_reranker
CACHE_MAX = settings.cache_max
CACHE_TTL = settings.cache_ttl
RAW_DIR = settings.raw_dir
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap
STAGING_DIR = settings.staging_dir
WHITELIST_PATH = settings.whitelist_path
SOURCE_CATALOG_PATH = settings.source_catalog_path
INGEST_CACHE_DIR = settings.ingest_cache_dir
OCR_MIN_CONFIDENCE = settings.ocr_min_confidence
OFFLINE_ONLY = settings.offline_only
PATCHES_DIR = settings.patches_dir
BACKUPS_DIR = settings.backups_dir
ADMIN_TOKEN = settings.admin_token
PATCH_PUBLIC_KEY_PATH = settings.patch_public_key_path
PATCH_SIGNING_KEY_PATH = settings.patch_signing_key
ALLOW_UNSIGNED_PATCHES = settings.allow_unsigned_patches
MAX_PATCH_BYTES = settings.max_patch_bytes
MAX_PATCH_UPLOAD_BYTES = settings.max_patch_upload_bytes
MAX_HISTORY = settings.max_history
MAX_SESSIONS = settings.max_sessions
SESSION_TTL = settings.session_ttl
API_HOST = settings.api_host
API_PORT = settings.api_port
API_RELOAD = settings.api_reload
API_CORS_ORIGINS: str = settings.cors_origins or ""
RATE_LIMIT_PER_MINUTE = settings.rate_limit_per_minute
MAX_QUESTION_CHARS = settings.max_question_chars
LOG_LEVEL = settings.log_level
LOG_FORMAT = settings.log_format

# Fixed values (not configurable)
SKIP_EXTENSIONS = {
    ".bak", ".jpg", ".jpeg", ".png", ".gif", ".bmp",
    ".zip", ".rar", ".mp4", ".mp3", ".wav",
}
SOURCE_TIERS = {"tier_1", "tier_2", "tier_3", "unverified"}
SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf", ".docx": "docx", ".doc": "doc",
    ".pptx": "pptx", ".pptm": "pptx", ".ppt": "ppt",
    ".txt": "txt", ".json": "json", ".csv": "csv", ".md": "markdown",
}

warn_unknown_variables()


def _main(argv=None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Show Vet AI Tutor settings.")
    parser.add_argument("--env-example", action="store_true", help="print a documented .env template")
    args = parser.parse_args(argv)
    if args.env_example:
        print(env_example())
        return
    rows = describe(settings)
    width = max(len(r[0]) for r in rows)
    for name, value, source in rows:
        print(f"{name:<{width}}  {value:<40}  [{source}]")
    for name, suggestion in unknown_variables().items():
        print(f"WARNING: unknown {name}" + (f" (did you mean {suggestion}?)" if suggestion else ""),
              file=sys.stderr)


if __name__ == "__main__":
    _main()

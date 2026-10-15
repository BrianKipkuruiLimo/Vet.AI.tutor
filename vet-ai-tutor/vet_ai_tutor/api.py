"""FastAPI production API for Vet AI Tutor.

Exposes the Tutor's RAG pipeline over HTTP with:
- Request ID tracking for observability
- Structured error responses
- Request timing middleware
- Static file serving for the chat UI
- Health endpoint with system diagnostics

Run with:
    uvicorn vet_ai_tutor.api:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import secrets
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from vet_ai_tutor.config import (
    VECTOR_PATH, LLM_MODEL, RETRIEVER_K, API_HOST, API_PORT,
    API_CORS_ORIGINS, VECTOR_BACKEND, EMBEDDING_MODEL, EMBEDDING_BACKEND,
    STAGING_DIR,
    ADMIN_TOKEN,
    OFFLINE_ONLY,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global tutor instance (initialized at startup)
# ---------------------------------------------------------------------------
_tutor = None
_start_time = None


def _init_tutor():
    """Initialize the Tutor with Ollama LLM and vectorstore."""
    global _tutor, _start_time
    from langchain_ollama import OllamaLLM  # pyrefly: ignore [missing-import]
    from vet_ai_tutor.core import Tutor
    from vet_ai_tutor.llm_adapter import OllamaAdapter

    base_llm = OllamaLLM(
        model=LLM_MODEL,
        num_predict=200,       # ~120 words max — keeps generation under 15s on CPU
        timeout=120,
        keep_alive="30m",
        num_ctx=1024,          # tight context window — halves KV cache RAM
    )
    adapter = OllamaAdapter(base_llm)
    _tutor = Tutor(llm=adapter, vector_path=VECTOR_PATH)

    # Pre-warm the model in Ollama memory
    try:
        base_llm.invoke("Hi")
    except Exception:
        pass

    if _tutor.db is None:
        raise RuntimeError(
            "Vectorstore not loaded. Run `python scripts/rebuild_vectorstore.py` first."
        )

    _start_time = time.time()
    logger.info(
        "Tutor initialized: model=%s, backend=%s, docs=%d",
        LLM_MODEL, _tutor.backend, _tutor.doc_count,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle for the FastAPI app."""
    logger.info("Starting Vet AI Tutor API...")
    _init_tutor()
    logger.info("Vet AI Tutor API ready.")
    yield
    logger.info("Shutting down Vet AI Tutor API.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Vet AI Tutor API",
    description=(
        "Production REST API for the Veterinary AI Tutor. "
        "Uses hybrid BM25 + vector search with optional cross-encoder re-ranking, "
        "LRU query caching, and streaming LLM inference via Ollama."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — configurable via VET_CORS_ORIGINS env var
_origins = [o.strip() for o in API_CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware: request ID + timing + structured error handling
# ---------------------------------------------------------------------------
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Add request ID, measure timing, and catch unhandled exceptions."""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    t0 = time.time()

    try:
        response = await call_next(request)
        elapsed = time.time() - t0
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{elapsed:.3f}s"

        # Log non-static requests
        if not request.url.path.startswith("/static"):
            logger.info(
                "[%s] %s %s → %d (%.2fs)",
                request_id, request.method, request.url.path,
                response.status_code, elapsed,
            )
        return response
    except Exception as e:
        elapsed = time.time() - t0
        logger.exception("[%s] Unhandled error on %s: %s", request_id, request.url.path, e)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "detail": str(e),
                "request_id": request_id,
            },
            headers={"X-Request-ID": request_id, "X-Response-Time": f"{elapsed:.3f}s"},
        )


# ---------------------------------------------------------------------------
# Static files & root
# ---------------------------------------------------------------------------
import os
_static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/", include_in_schema=False)
async def root():
    """Serve the chat UI if available, otherwise redirect to API docs."""
    chat_path = os.path.join(_static_dir, "chat.html")
    if os.path.exists(chat_path):
        return FileResponse(chat_path)
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The veterinary question to ask.")
    k: Optional[int] = Field(None, ge=1, le=20, description="Number of documents to retrieve.")
    remember: bool = Field(True, description="Whether to store this exchange in conversation history.")


class Citation(BaseModel):
    id: str
    target: str


class AskResponse(BaseModel):
    answer: str
    sources: List[dict] = Field(default_factory=list, description="Retrieved source documents.")
    citations: List[Citation] = Field(default_factory=list)


class HistoryEntry(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    history: List[HistoryEntry]
    count: int


class FollowupRequest(BaseModel):
    question: str
    answer: str
    num: int = Field(3, ge=1, le=5)


class FollowupResponse(BaseModel):
    followups: List[str]


class HealthResponse(BaseModel):
    status: str
    model: str
    embedding_model: str
    embedding_backend: str
    vector_backend: str
    vector_path: str
    vectorstore_loaded: bool
    doc_count: int
    cache_stats: dict
    uptime_seconds: Optional[float] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Health check — reports system status, doc count, cache stats, and uptime."""
    uptime = time.time() - _start_time if _start_time else None
    return HealthResponse(
        status="ok" if _tutor and _tutor.db else "degraded",
        model=LLM_MODEL,
        embedding_model=EMBEDDING_MODEL,
        embedding_backend=EMBEDDING_BACKEND,
        vector_backend=VECTOR_BACKEND,
        vector_path=VECTOR_PATH,
        vectorstore_loaded=_tutor is not None and _tutor.db is not None,
        doc_count=_tutor.doc_count if _tutor else 0,
        cache_stats=_tutor.cache_stats if _tutor else {},
        uptime_seconds=round(uptime, 1) if uptime else None,
    )


@app.post("/ask", response_model=AskResponse, tags=["tutor"])
async def ask(req: AskRequest):
    """Ask a veterinary question and receive a complete answer with sources."""
    if _tutor is None or _tutor.db is None:
        raise HTTPException(status_code=503, detail="Tutor not initialized.")

    try:
        answer, docs = _tutor.ask(
            question=req.question,
            k=req.k or RETRIEVER_K,
            remember=req.remember,
        )
    except Exception as e:
        logger.exception("Error during ask: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    sources = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        sources.append({
            "source": meta.get("source", meta.get("source_file", "unknown")),
            "chunk": meta.get("chunk", meta.get("chunk_index")),
            "species": meta.get("species", ""),
            "topic": meta.get("topic", ""),
            "source_tier": meta.get("source_tier", ""),
            "source_url": meta.get("source_url", ""),
            "source_title": meta.get("source_title", ""),
            "author": meta.get("author", ""),
            "journal": meta.get("journal", ""),
            "publication_year": meta.get("publication_year", ""),
            "page": meta.get("page", ""),
            "slide": meta.get("slide", ""),
            "content_preview": (getattr(d, "page_content", "") or "")[:200],
        })

    return AskResponse(answer=answer, sources=sources)


@app.post("/ask_stream", tags=["tutor"])
async def ask_stream(req: AskRequest):
    """Ask a veterinary question and receive a streamed answer (SSE-style)."""
    if _tutor is None or _tutor.db is None:
        raise HTTPException(status_code=503, detail="Tutor not initialized.")

    def _generate():
        try:
            for token, docs in _tutor.ask_stream(
                question=req.question,
                k=req.k or RETRIEVER_K,
                remember=req.remember,
            ):
                yield token
        except Exception as e:
            logger.exception("Streaming error: %s", e)
            yield f"\n\n[Error: {e}]"

    return StreamingResponse(_generate(), media_type="text/plain")


@app.get("/history", response_model=HistoryResponse, tags=["session"])
async def get_history():
    """Retrieve the current conversation history."""
    if _tutor is None:
        raise HTTPException(status_code=503, detail="Tutor not initialized.")

    history = _tutor.get_history()
    entries = [HistoryEntry(role=role, content=content) for role, content in history]
    return HistoryResponse(history=entries, count=len(entries))


@app.post("/reset", tags=["session"])
async def reset():
    """Reset the conversation history and query cache."""
    if _tutor is None:
        raise HTTPException(status_code=503, detail="Tutor not initialized.")

    _tutor.reset_history()
    return {"status": "ok", "message": "Conversation history and cache cleared."}


@app.post("/followups", response_model=FollowupResponse, tags=["tutor"])
async def followups(req: FollowupRequest):
    """Generate suggested follow-up questions based on a Q&A pair."""
    if _tutor is None:
        raise HTTPException(status_code=503, detail="Tutor not initialized.")

    try:
        suggestions = _tutor.suggest_followups(req.question, req.answer, num=req.num)
    except Exception as e:
        logger.exception("Error generating follow-ups: %s", e)
        suggestions = []

    return FollowupResponse(followups=suggestions)


# ---------------------------------------------------------------------------
# Developer-only endpoints (ingestion pipeline management)
# ---------------------------------------------------------------------------

def _project_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(os.path.dirname(os.path.dirname(__file__)), path)


async def require_admin(x_admin_token: str | None = Header(default=None)):
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="Developer updates are disabled; configure VET_ADMIN_TOKEN.")
    if not x_admin_token or not secrets.compare_digest(x_admin_token, ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="A valid developer token is required.")

class StagingFile(BaseModel):
    filename: str
    species: str
    category: str
    source_url: str
    source_tier: str
    status: str
    date: str


class StagingListResponse(BaseModel):
    files: List[StagingFile]
    total: int


@app.get("/dev/staging", response_model=StagingListResponse, tags=["developer"])
async def list_staging(_auth=Depends(require_admin)):
    """List all files in the staging area (developer-only)."""
    import glob

    staging_path = _project_path(STAGING_DIR)
    if not os.path.exists(staging_path):
        return StagingListResponse(files=[], total=0)

    files = []
    for fp in sorted(glob.glob(os.path.join(staging_path, "*.json"))):
        basename = os.path.basename(fp)
        if basename in ("harvest_log.json", "ingest_log.json"):
            continue
        try:
            import json as json_mod
            with open(fp, "r", encoding="utf-8") as f:
                doc = json_mod.load(f)
            files.append(StagingFile(
                filename=basename,
                species=doc.get("species", "?"),
                category=doc.get("category", "?"),
                source_url=doc.get("source_url", "?"),
                source_tier=doc.get("source_tier", "?"),
                status=doc.get("status", "unknown"),
                date=doc.get("date_harvested", "?"),
            ))
        except Exception:
            continue

    return StagingListResponse(files=files, total=len(files))


@app.get("/dev/health", tags=["developer"])
async def dev_health(_auth=Depends(require_admin)):
    """Extended health check with ingestion pipeline status."""
    import glob

    staging_path = _project_path(STAGING_DIR)
    staging_count = 0
    if os.path.exists(staging_path):
        staging_count = len([
            f for f in glob.glob(os.path.join(staging_path, "*.json"))
            if os.path.basename(f) not in ("harvest_log.json", "ingest_log.json")
        ])

    return {
        "status": "ok" if _tutor and _tutor.db else "degraded",
        "model": LLM_MODEL,
        "doc_count": _tutor.doc_count if _tutor else 0,
        "staging_files": staging_count,
        "pipeline": {
            "harvester": "scripts/harvester.py",
            "document_ingestor": "scripts/ingest_documents.py",
            "staged_ingestor": "scripts/ingest_staged.py",
            "patch_creator": "scripts/create_data_patch.py",
            "patch_applier": "scripts/apply_data_patch.py",
        },
    }


@app.post("/knowledge/update", tags=["developer"], dependencies=[Depends(require_admin)])
async def update_knowledge(patch: UploadFile = File(...)):
    """Validate and atomically apply a developer-supplied knowledge patch."""
    limit = int(os.environ.get("VET_MAX_PATCH_UPLOAD_BYTES", str(8 * 1024**3)))
    applied = None
    try:
        with tempfile.NamedTemporaryFile(prefix="vet-patch-", suffix=".zip", delete=False) as temp:
            temp_path = temp.name
            size = 0
            while block := await patch.read(1024 * 1024):
                size += len(block)
                if size > limit:
                    raise HTTPException(status_code=413, detail="Patch exceeds upload size limit.")
                temp.write(block)
        from scripts.apply_data_patch import apply_patch
        result = apply_patch(temp_path)
        applied = result
        _init_tutor()
        return {"status": "ok", "version": result["version"], "message": "Knowledge base updated."}
    except HTTPException:
        raise
    except Exception as exc:
        if applied and applied.get("backup_path"):
            try:
                import shutil
                import tempfile as tempfile_mod
                target = os.path.abspath(VECTOR_PATH)
                backup = applied["backup_path"]
                rollback_stage = tempfile_mod.mkdtemp(prefix=".vet-rollback-", dir=os.path.dirname(target))
                shutil.rmtree(rollback_stage)
                shutil.copytree(backup, rollback_stage)
                failed_index = target + ".failed-update"
                if os.path.exists(failed_index):
                    shutil.rmtree(failed_index)
                if os.path.exists(target):
                    os.replace(target, failed_index)
                os.replace(rollback_stage, target)
                shutil.rmtree(failed_index, ignore_errors=True)
                _init_tutor()
            except Exception:
                logger.exception("Could not restore the previous index after failed reload")
        logger.exception("Knowledge patch failed")
        raise HTTPException(status_code=400, detail=f"Patch rejected: {exc}")
    finally:
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    uvicorn.run(
        "vet_ai_tutor.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )

"""FastAPI production API for Vet AI Tutor.

Exposes the Tutor's RAG pipeline over HTTP with:
- Per-session conversation history (clients send a `session_id`)
- Server-Sent Events streaming that also delivers sources
- Request ID tracking and timing headers
- One error envelope: {"error", "detail", "request_id"}
- Per-client rate limiting on question endpoints
- Static file serving for the chat UI

Run with:
    python -m vet_ai_tutor.api
"""

import json
import logging
import os
import re
import secrets
import tempfile
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any, List, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from vet_ai_tutor.config import (
    ADMIN_TOKEN, API_CORS_ORIGINS, API_HOST, API_PORT, API_RELOAD, EMBEDDING_BACKEND,
    MAX_PATCH_UPLOAD_BYTES,
    EMBEDDING_MODEL, LLM_MODEL, MAX_QUESTION_CHARS, RATE_LIMIT_PER_MINUTE,
    RETRIEVER_K, STAGING_DIR, VECTOR_BACKEND, VECTOR_PATH, project_path,
)
from vet_ai_tutor.observability import configure_logging, metrics, request_id_var

logger = logging.getLogger(__name__)

SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{8,64}$"

# ---------------------------------------------------------------------------
# Tutor lifecycle
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {"tutor": None, "start_time": None, "init_error": None, "quiz": None}
_init_lock = threading.Lock()


def _init_tutor():
    """Initialize the Tutor with the configured Ollama LLM and vectorstore."""
    from vet_ai_tutor.core import Tutor
    from vet_ai_tutor.llm_adapter import build_llm
    from vet_ai_tutor.quiz import QuizBank

    with _init_lock:
        _state["quiz"] = QuizBank()  # reloaded with the index after knowledge updates
        llm = build_llm()
        tutor = Tutor(llm=llm, vector_path=VECTOR_PATH)
        llm.warm_up()
        if tutor.db is None:
            raise RuntimeError(
                "Vectorstore not loaded. Run `python scripts/rebuild_vectorstore.py` first."
            )
        _state.update(tutor=tutor, start_time=time.time(), init_error=None)
        logger.info(
            "Tutor initialized: model=%s, backend=%s, docs=%d",
            LLM_MODEL, tutor.backend, tutor.doc_count,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start in degraded mode instead of crashing, so /health can explain why."""
    configure_logging()
    logger.info("Starting Vet AI Tutor API...")
    try:
        await run_in_threadpool(_init_tutor)
        logger.info("Vet AI Tutor API ready.")
    except Exception as e:
        _state["init_error"] = str(e)
        logger.exception("Tutor failed to initialize; serving in degraded mode.")
    yield
    logger.info("Shutting down Vet AI Tutor API.")


def get_tutor():
    """Dependency: the ready Tutor, or 503 if it is not loaded."""
    tutor = _state["tutor"]
    if tutor is None or tutor.db is None:
        detail = _state["init_error"] or "Tutor not initialized."
        raise HTTPException(status_code=503, detail=detail)
    return tutor


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Vet AI Tutor API",
    description=(
        "REST API for the Veterinary AI Tutor. Uses hybrid BM25 + vector search "
        "fused with Reciprocal Rank Fusion, LRU query caching, and streaming "
        "LLM inference via a local Ollama model."
    ),
    version="2.1.0",
    lifespan=lifespan,
)

_origins = [o.strip() for o in API_CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    # Credentials with a wildcard origin would let any site act for the user.
    allow_credentials="*" not in _origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Admin-Token"],
)


# ---------------------------------------------------------------------------
# Error envelope
# ---------------------------------------------------------------------------
def _error(request: Request, status: int, error: str, detail, headers=None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": error, "detail": detail,
                 "request_id": getattr(request.state, "request_id", None)},
        headers=headers,
    )


_ERROR_CODES = {400: "bad_request", 401: "unauthorized", 404: "not_found",
                413: "payload_too_large", 429: "rate_limited", 503: "unavailable"}


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    code = _ERROR_CODES.get(exc.status_code, "http_error")
    return _error(request, exc.status_code, code, exc.detail, getattr(exc, "headers", None))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    detail = "; ".join(
        f"{'.'.join(str(p) for p in err['loc'][1:]) or 'body'}: {err['msg']}" for err in exc.errors()
    )
    return _error(request, 422, "validation_error", detail)


# ---------------------------------------------------------------------------
# Middleware: rate limit + request ID + timing
# ---------------------------------------------------------------------------
_RATE_LIMITED_PATHS = {"/ask", "/ask_stream", "/followups"}
_request_log: dict = defaultdict(deque)
_rate_lock = threading.Lock()


def _rate_limited(client: str) -> bool:
    if RATE_LIMIT_PER_MINUTE <= 0:
        return False
    now = time.monotonic()
    with _rate_lock:
        window = _request_log[client]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= RATE_LIMIT_PER_MINUTE:
            return True
        window.append(now)
        return False


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Add request ID, enforce the rate limit, measure timing, catch errors."""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    request_id_var.set(request_id)  # attached to every log line of this request
    t0 = time.time()

    if request.url.path in _RATE_LIMITED_PATHS and request.method == "POST":
        client = request.client.host if request.client else "unknown"
        if _rate_limited(client):
            return _error(request, 429, "rate_limited",
                          f"Limit is {RATE_LIMIT_PER_MINUTE} questions per minute.",
                          {"Retry-After": "60", "X-Request-ID": request_id})

    try:
        response = await call_next(request)
    except Exception:
        logger.exception("[%s] Unhandled error on %s", request_id, request.url.path)
        response = _error(request, 500, "internal_server_error",
                          "Unexpected server error; see the server log.")
    elapsed = time.time() - t0
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{elapsed:.3f}s"
    if not request.url.path.startswith("/static"):
        # A streamed answer is still being generated here; its timing is in the "question" line.
        streamed = response.headers.get("content-type", "").startswith("text/event-stream")
        logger.info("%s %s → %d (%s%.2fs)", request.method, request.url.path, response.status_code,
                    "stream started in " if streamed else "", elapsed,
                    extra={"event": "request", "method": request.method, "path": request.url.path,
                           "status": response.status_code, "seconds": round(elapsed, 4),
                           "streamed": streamed})
    return response


# ---------------------------------------------------------------------------
# Static files & root
# ---------------------------------------------------------------------------
_static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/", include_in_schema=False)
async def root():
    """Serve the chat UI if available, otherwise redirect to API docs."""
    chat_path = os.path.join(_static_dir, "chat.html")
    if os.path.exists(chat_path):
        return FileResponse(chat_path)
    return RedirectResponse(url="/docs")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS,
                          description="The veterinary question to ask.")
    k: Optional[int] = Field(None, ge=1, le=20, description="Number of documents to retrieve.")
    remember: bool = Field(True, description="Whether to store this exchange in conversation history.")
    session_id: str = Field("default", pattern=SESSION_ID_PATTERN,
                            description="Conversation ID; each browser tab should use its own.")


class Citation(BaseModel):
    id: str
    target: str


class AskResponse(BaseModel):
    answer: str
    sources: List[dict] = Field(default_factory=list, description="Retrieved source documents.")
    citations: List[Citation] = Field(default_factory=list)
    session_id: str


class HistoryEntry(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    history: List[HistoryEntry]
    count: int


class ResetRequest(BaseModel):
    session_id: str = Field("default", pattern=SESSION_ID_PATTERN)


class FollowupRequest(BaseModel):
    question: str = Field(..., max_length=MAX_QUESTION_CHARS)
    answer: str = Field(..., max_length=4000)
    num: int = Field(3, ge=1, le=5)


class FollowupResponse(BaseModel):
    followups: List[str]


class HealthResponse(BaseModel):
    status: str
    model: str
    embedding_model: str
    embedding_backend: str
    vector_backend: str
    vectorstore_loaded: bool
    doc_count: int
    cache_stats: dict
    active_sessions: int = 0
    uptime_seconds: Optional[float] = None
    warning: Optional[str] = None
    performance: dict = Field(default_factory=dict, description="Median seconds per stage")


def _source_entry(d) -> dict:
    meta = getattr(d, "metadata", {}) or {}
    return {
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
    }


def _citations(docs) -> List[Citation]:
    return [Citation(id=f"[S{i}]", target=s["source"] if s["chunk"] is None else f"{s['source']}#{s['chunk']}")
            for i, s in enumerate((_source_entry(d) for d in docs), start=1)]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Health check — reports system status, doc count, cache stats, and uptime."""
    tutor = _state["tutor"]
    loaded = tutor is not None and tutor.db is not None
    start = _state["start_time"]
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model=LLM_MODEL,
        embedding_model=EMBEDDING_MODEL,
        embedding_backend=EMBEDDING_BACKEND,
        vector_backend=VECTOR_BACKEND,
        vectorstore_loaded=loaded,
        doc_count=tutor.doc_count if tutor else 0,
        cache_stats=tutor.cache_stats if tutor else {},
        active_sessions=tutor.session_count if tutor else 0,
        uptime_seconds=round(time.time() - start, 1) if start else None,
        warning=_state["init_error"] or (tutor.index_warning if tutor else None),
        performance=_performance_summary(),
    )


def _performance_summary() -> dict:
    """Small public summary; the full breakdown is on /dev/metrics."""
    summary = metrics.summary()
    medians = {name: s["p50"] for name, s in summary["stages"].items()
               if name in ("total_seconds", "first_token", "generation", "tokens_per_second")}
    return {"questions": summary["questions"], "median": medians} if summary["questions"] else {}


# Question endpoints are plain `def`: FastAPI runs them in a worker thread, so
# a slow CPU generation does not block the event loop for other clients.
@app.post("/ask", response_model=AskResponse, tags=["tutor"])
def ask(req: AskRequest, tutor=Depends(get_tutor)):
    """Ask a veterinary question and receive a complete answer with sources."""
    answer, docs = tutor.ask(
        question=req.question, k=req.k or RETRIEVER_K,
        remember=req.remember, session_id=req.session_id,
    )
    return AskResponse(answer=answer, sources=[_source_entry(d) for d in docs],
                       citations=_citations(docs), session_id=req.session_id)


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/ask_stream", tags=["tutor"])
def ask_stream(req: AskRequest, tutor=Depends(get_tutor)):
    """Stream an answer as Server-Sent Events.

    Events: `sources` (list, sent once before the answer), `token` (string),
    `error` ({"detail"}), and `done` ({}).
    """
    def _generate():
        sent_sources = False
        try:
            for token, docs in tutor.ask_stream(
                question=req.question, k=req.k or RETRIEVER_K,
                remember=req.remember, session_id=req.session_id,
            ):
                if not sent_sources:
                    yield _sse("sources", [_source_entry(d) for d in docs])
                    sent_sources = True
                yield _sse("token", token)
        except Exception:
            logger.exception("Streaming error")
            yield _sse("error", {"detail": "The tutor failed while answering; see the server log."})
        yield _sse("done", {})

    return StreamingResponse(_generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/history", response_model=HistoryResponse, tags=["session"])
def get_history(session_id: str = "default", tutor=Depends(get_tutor)):
    """Retrieve one session's conversation history."""
    if not _valid_session(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id.")
    entries = [HistoryEntry(role=role, content=content)
               for role, content in tutor.get_history(session_id)]
    return HistoryResponse(history=entries, count=len(entries))


@app.post("/reset", tags=["session"])
def reset(req: Optional[ResetRequest] = None, tutor=Depends(get_tutor)):
    """Clear one session's conversation history."""
    session_id = req.session_id if req else "default"
    tutor.reset_history(session_id)
    return {"status": "ok", "message": "Conversation history cleared.", "session_id": session_id}


@app.post("/followups", response_model=FollowupResponse, tags=["tutor"])
def followups(req: FollowupRequest, tutor=Depends(get_tutor)):
    """Generate suggested follow-up questions based on a Q&A pair."""
    try:
        suggestions = tutor.suggest_followups(req.question, req.answer, num=req.num)
    except Exception:
        logger.exception("Error generating follow-ups")
        suggestions = []
    return FollowupResponse(followups=suggestions)


# ---------------------------------------------------------------------------
# Quiz mode (reviewed question bank shipped with the knowledge index)
# ---------------------------------------------------------------------------

class QuizQuestion(BaseModel):
    id: str
    question: str
    options: List[str]


class QuizResponse(BaseModel):
    questions: List[QuizQuestion]
    available: int


class QuizCheckRequest(BaseModel):
    id: str = Field(..., max_length=64)
    choice: int = Field(..., ge=0, le=3)


def get_quiz():
    from vet_ai_tutor.quiz import QuizBank
    if _state["quiz"] is None:
        _state["quiz"] = QuizBank()
    return _state["quiz"]


@app.get("/quiz", response_model=QuizResponse, tags=["quiz"])
def quiz(n: int = 5, topic: str = "", bank=Depends(get_quiz)):
    """Random reviewed questions (answers are not included)."""
    if not 1 <= n <= 20 or len(topic) > 100:
        raise HTTPException(status_code=400, detail="n must be 1-20 and topic at most 100 characters.")
    return QuizResponse(questions=[QuizQuestion(**q) for q in bank.pick(n, topic)], available=len(bank))


@app.get("/quiz/topics", tags=["quiz"])
def quiz_topics(bank=Depends(get_quiz)):
    """Source documents that have quiz questions, with counts."""
    return {"topics": bank.topics(), "total": len(bank)}


@app.post("/quiz/check", tags=["quiz"])
def quiz_check(req: QuizCheckRequest, bank=Depends(get_quiz)):
    """Grade one answer; returns the correct option, explanation and source."""
    result = bank.check(req.id, req.choice)
    if result is None:
        raise HTTPException(status_code=404, detail="Unknown question id.")
    return result


def _valid_session(session_id: str) -> bool:
    return re.fullmatch(SESSION_ID_PATTERN, session_id) is not None


# ---------------------------------------------------------------------------
# Developer-only endpoints (ingestion pipeline management)
# ---------------------------------------------------------------------------

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


_STAGING_LOGS = ("harvest_log.json", "ingest_log.json", "ingested_hashes.json")


def _staging_records() -> List[str]:
    import glob
    staging_path = project_path(STAGING_DIR)
    if not os.path.exists(staging_path):
        return []
    return [fp for fp in sorted(glob.glob(os.path.join(staging_path, "*.json")))
            if os.path.basename(fp) not in _STAGING_LOGS]


@app.get("/dev/staging", response_model=StagingListResponse, tags=["developer"])
def list_staging(_auth=Depends(require_admin)):
    """List all files in the staging area (developer-only)."""
    files = []
    for fp in _staging_records():
        try:
            with open(fp, encoding="utf-8") as f:
                doc = json.load(f)
            files.append(StagingFile(
                filename=os.path.basename(fp),
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
def dev_health(_auth=Depends(require_admin)):
    """Extended health check with ingestion pipeline status."""
    tutor = _state["tutor"]
    return {
        "status": "ok" if tutor and tutor.db else "degraded",
        "model": LLM_MODEL,
        "doc_count": tutor.doc_count if tutor else 0,
        "staging_files": len(_staging_records()),
        "index_warning": tutor.index_warning if tutor else None,
        "pipeline": {
            "harvester": "scripts/harvester.py",
            "document_ingestor": "scripts/ingest_documents.py",
            "staged_ingestor": "scripts/ingest_staged.py",
            "patch_creator": "scripts/create_data_patch.py",
            "patch_applier": "scripts/apply_data_patch.py",
        },
    }


@app.get("/dev/metrics", tags=["developer"], dependencies=[Depends(require_admin)])
def dev_metrics():
    """Per-stage timings (median, 95th percentile, max) over recent questions."""
    return metrics.summary()


@app.post("/dev/cache/clear", tags=["developer"], dependencies=[Depends(require_admin)])
def clear_cache(tutor=Depends(get_tutor)):
    """Drop all cached answers (e.g. after changing prompts)."""
    tutor.clear_cache()
    return {"status": "ok"}


def _restore_backup(backup: str) -> None:
    """Put the pre-update index back after the new one failed to load."""
    import shutil
    target = os.path.abspath(VECTOR_PATH)
    rollback_stage = tempfile.mkdtemp(prefix=".vet-rollback-", dir=os.path.dirname(target))
    shutil.rmtree(rollback_stage)
    shutil.copytree(backup, rollback_stage)
    failed_index = target + ".failed-update"
    if os.path.exists(failed_index):
        shutil.rmtree(failed_index)
    if os.path.exists(target):
        os.replace(target, failed_index)
    os.replace(rollback_stage, target)
    shutil.rmtree(failed_index, ignore_errors=True)


@app.post("/knowledge/update", tags=["developer"], dependencies=[Depends(require_admin)])
async def update_knowledge(patch: UploadFile = File(...)):
    """Verify, validate and atomically apply a signed knowledge patch."""
    from scripts.apply_data_patch import apply_patch

    limit = MAX_PATCH_UPLOAD_BYTES
    temp_path = None
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
        applied = await run_in_threadpool(apply_patch, temp_path)
        await run_in_threadpool(_init_tutor)
        return {"status": "ok", "version": applied["version"], "message": "Knowledge base updated."}
    except HTTPException:
        raise
    except Exception as exc:
        if applied and applied.get("backup_path"):
            try:
                await run_in_threadpool(_restore_backup, applied["backup_path"])
                await run_in_threadpool(_init_tutor)
            except Exception:
                logger.exception("Could not restore the previous index after failed reload")
        logger.exception("Knowledge patch failed")
        raise HTTPException(status_code=400, detail=f"Patch rejected: {exc}") from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    configure_logging()
    uvicorn.run(
        "vet_ai_tutor.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD,
    )

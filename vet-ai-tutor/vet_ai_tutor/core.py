import os
import re
import time
import logging
import hashlib
import threading
from typing import List, Tuple, Optional
from collections import OrderedDict

import numpy as np

from vet_ai_tutor.config import (
    VECTOR_PATH, EMBEDDING_MODEL, EMBEDDING_BACKEND,
    RETRIEVER_K, MAX_HISTORY, VECTOR_BACKEND,
)
 
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LRU Cache for query results — avoids redundant LLM calls for repeated
# questions. Thread-safe via lock. TTL prevents stale answers.
# ---------------------------------------------------------------------------
_CACHE_MAX = int(os.environ.get("VET_CACHE_MAX", "128"))
_CACHE_TTL = int(os.environ.get("VET_CACHE_TTL", "3600"))  # seconds


class _QueryCache:
    """Thread-safe LRU cache with TTL expiration for query results."""

    def __init__(self, maxsize: int = _CACHE_MAX, ttl: int = _CACHE_TTL):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _key(self, question: str, k: int) -> str:
        return hashlib.md5(f"{question.strip().lower()}:k={k}".encode()).hexdigest()

    def get(self, question: str, k: int) -> Optional[tuple]:
        key = self._key(question, k)
        with self._lock:
            if key in self._cache:
                answer, docs, ts = self._cache[key]
                if time.time() - ts < self._ttl:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return answer, docs
                else:
                    del self._cache[key]
            self._misses += 1
            return None

    def put(self, question: str, k: int, answer: str, docs: list):
        key = self._key(question, k)
        with self._lock:
            self._cache[key] = (answer, docs, time.time())
            self._cache.move_to_end(key)
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    @property
    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{self._hits / total:.1%}" if total else "N/A",
            }

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0


# ---------------------------------------------------------------------------
# Optional cross-encoder re-ranker — disabled by default to save RAM & startup
# time. Enable by setting VET_USE_RERANKER=1 in environment.
# ---------------------------------------------------------------------------
_reranker = None
if os.environ.get("VET_USE_RERANKER", "0") == "1":
    try:
        # pyrefly: ignore [missing-import]
        from sentence_transformers import CrossEncoder
        _model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        try:
            _reranker = CrossEncoder(_model_name, local_files_only=True)
            logger.info("Cross-encoder re-ranker loaded from local cache.")
        except Exception:
            logger.info(
                "Re-ranker model '%s' not in local cache; skipping. "
                "Pull it first with internet access.", _model_name
            )
    except ImportError:
        logger.info("sentence_transformers not installed; re-ranking disabled.")


def _build_bm25_retriever(docs, k: int):
    """Build a BM25 retriever from a list of LangChain Documents."""
    try:
        # pyrefly: ignore [missing-import]
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank_bm25 not installed; hybrid search will use vector-only.")
        return None

    corpus = [
        (getattr(d, "page_content", "") or "").lower().split()
        for d in docs
    ]
    if not corpus:
        return None

    return BM25Okapi(corpus), docs


# ---------------------------------------------------------------------------
# Greeting pattern (compiled once)
# ---------------------------------------------------------------------------
_GREETING_RE = re.compile(
    r"^(hi|hello|hey|good\s(morning|afternoon|evening))\b", re.IGNORECASE
)

# ---------------------------------------------------------------------------
# System prompt — kept SHORT to minimize input tokens on CPU inference.
# Every extra token in the prompt = slower generation on low-end hardware.
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT_RAG = (
    "You are a veterinary tutor. Use ONLY the provided Context to answer. "
    "Include the specific terms, names, and classifications from the Context. "
    "Cite sources as [S1],[S2]. Keep answer under 150 words. "
    "If the answer is not in the Context, say \"Not found in my knowledge base.\"\n"
)

_SYSTEM_PROMPT_CHAT = (
    "Veterinary tutor. Answer briefly. Say \"I don't know\" if unsure.\n"
)


def _load_embeddings(embedding_model: str, embedding_backend: str):
    """Load an embedding function — Ollama first (zero PyTorch), HuggingFace fallback.

    Using Ollama embeddings means no sentence-transformers / PyTorch required,
    saving ~3-4 GB of disk space and significant startup time.
    """
    from vet_ai_tutor.config import OFFLINE_ONLY
    if embedding_backend == "ollama":
        try:
            # pyrefly: ignore [missing-import]
            from langchain_ollama import OllamaEmbeddings
            emb = OllamaEmbeddings(model=embedding_model)
            # Quick smoke-test — if Ollama is not running this raises immediately
            emb.embed_query("test")
            logger.info("OllamaEmbeddings loaded: model=%s", embedding_model)
            return emb
        except Exception as e:
            logger.warning(
                "OllamaEmbeddings failed (is Ollama running? did you run "
                "'ollama pull %s'?): %s — falling back to HuggingFace.",
                embedding_model, e,
            )

    if OFFLINE_ONLY and embedding_backend != "huggingface":
        logger.error("Ollama embeddings unavailable and offline mode is enabled; refusing network fallback.")
        return None

    # HuggingFace fallback is only available when offline-only mode is disabled.
    HFEmbeddings = None
    try:
        # pyrefly: ignore [missing-import]
        from langchain_huggingface import HuggingFaceEmbeddings as HFEmbeddings
    except ImportError:
        try:
            # pyrefly: ignore [missing-import]
            from langchain_community.embeddings import HuggingFaceEmbeddings as HFEmbeddings
        except ImportError:
            pass

    if HFEmbeddings is None:
        logger.error(
            "No embedding backend available. Install sentence-transformers or "
            "ensure Ollama is running."
        )
        return None

    # Use a lightweight model if falling back
    hf_model = embedding_model if embedding_backend == "huggingface" else "sentence-transformers/all-MiniLM-L6-v2"
    try:
        emb = HFEmbeddings(model_name=hf_model, model_kwargs={"local_files_only": True})
        logger.info("HuggingFaceEmbeddings loaded from cache: %s", hf_model)
        return emb
    except Exception:
        if OFFLINE_ONLY:
            logger.error("Offline mode: Hugging Face embedding model is not cached locally.")
            return None
        try:
            emb = HFEmbeddings(model_name=hf_model)
            logger.info("HuggingFaceEmbeddings downloaded: %s", hf_model)
            return emb
        except Exception as e:
            logger.error("HuggingFaceEmbeddings failed: %s", e)
            return None


class Tutor:
    """Production RAG tutor — centralizes embeddings, vectorstore, retriever and LLM.

    Features:
    - Hybrid retrieval (vector + BM25) with optional cross-encoder re-ranking
    - LRU query caching with TTL to avoid redundant LLM calls
    - Streaming support for low-latency UX
    - Thread-safe conversation history
    - Graceful degradation when dependencies are unavailable

    The `llm` passed in should expose an `invoke(prompt: str) -> str` method.
    """

    def __init__(self, llm, vector_path: str = None, embedding_model: str = None):
        self.vector_path = vector_path or VECTOR_PATH
        self.embedding_model = embedding_model or EMBEDDING_MODEL
        self.llm = llm
        self.history: List[tuple] = []
        self._history_lock = threading.Lock()
        self._cache = _QueryCache()

        os.makedirs(self.vector_path, exist_ok=True)

        # --- Load embeddings (Ollama first, HuggingFace fallback) ---
        t0 = time.time()
        self.embeddings = _load_embeddings(self.embedding_model, EMBEDDING_BACKEND)
        logger.info("Embeddings loaded in %.1fs", time.time() - t0)

        # --- Resolve vectorstore backend ---
        VectorStore = None
        self._backend = None

        if VECTOR_BACKEND == "chroma":
            try:
                # pyrefly: ignore [missing-import]
                from langchain_chroma import Chroma
                VectorStore = Chroma
                self._backend = "chroma"
            except Exception:
                logger.info("ChromaDB not installed; falling back to FAISS.")

        if VectorStore is None:
            try:
                # pyrefly: ignore [missing-import]
                from langchain_community.vectorstores import FAISS
                VectorStore = FAISS
                self._backend = "faiss"
            except Exception:
                logger.warning("No vectorstore backend available (neither ChromaDB nor FAISS).")

        if self.embeddings is None or VectorStore is None:
            self.db = None
            self._bm25_data = None
            self._doc_count = 0
            logger.warning("Vectorstore not loaded due to missing dependencies.")
            return

        # --- Load vectorstore ---
        t1 = time.time()
        try:
            if self._backend == "chroma":
                self.db = VectorStore(
                    persist_directory=self.vector_path,
                    embedding_function=self.embeddings,
                )
                count = self.db._collection.count()
                if count == 0:
                    logger.warning("ChromaDB collection is empty at %s", self.vector_path)
                    self.db = None
                    self._doc_count = 0
                else:
                    self._doc_count = count
                    logger.info("ChromaDB loaded: %d documents in %.1fs", count, time.time() - t1)
            else:  # faiss
                self.db = VectorStore.load_local(
                    self.vector_path, self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                self._doc_count = len(self.db.docstore._dict) if self.db else 0
                logger.info("FAISS loaded: %d documents in %.1fs", self._doc_count, time.time() - t1)
        except Exception as e:
            logger.exception("Failed to load %s vectorstore: %s", self._backend, e)
            self.db = None
            self._doc_count = 0

        # --- Build lightweight BM25 index for hybrid search ---
        self._bm25_data = None
        if self.db is not None:
            t2 = time.time()
            try:
                if self._backend == "chroma":
                    result = self.db.get()
                    if result and result.get("documents"):
                        from types import SimpleNamespace
                        all_docs = [
                            SimpleNamespace(
                                page_content=text,
                                metadata=(result["metadatas"][i] if result.get("metadatas") else {}),
                            )
                            for i, text in enumerate(result["documents"])
                        ]
                    else:
                        all_docs = []
                else:
                    all_docs = list(self.db.docstore._dict.values())

                if all_docs:
                    self._bm25_data = _build_bm25_retriever(all_docs, RETRIEVER_K)
                    if self._bm25_data:
                        logger.info("BM25 index built: %d documents in %.1fs", len(all_docs), time.time() - t2)
            except Exception as e:
                logger.debug("Could not build BM25 index: %s", e)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def doc_count(self) -> int:
        """Number of documents/chunks in the vectorstore."""
        return self._doc_count

    @property
    def backend(self) -> Optional[str]:
        """Active vectorstore backend name ('chroma' or 'faiss')."""
        return self._backend

    @property
    def cache_stats(self) -> dict:
        """Query cache statistics."""
        return self._cache.stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_greeting(self, question: str) -> str | None:
        if _GREETING_RE.match(question.strip()):
            return "Hello! I'm your Vet AI Tutor — ask me a veterinary question or say what you'd like to learn."
        return None

    def _enrich_query(self, question: str) -> str:
        """Prepend topic from history when the question is a short follow-up."""
        if len(question.split()) < 8 and self.history:
            prev_user_msgs = [c for r, c in self.history[:-1] if r == "user"]
            if prev_user_msgs:
                return prev_user_msgs[-1] + " " + question
        return question

    def _build_context(self, docs: List) -> Tuple[str, List[dict]]:
        """Build labeled context string and citation list from retrieved docs."""
        context_parts = []
        citations = []
        for i, d in enumerate(docs, start=1):
            meta = getattr(d, "metadata", {}) or {}
            source = meta.get("source") or meta.get("source_file") or meta.get("doc_id") or "unknown"
            chunk = meta.get("chunk") if "chunk" in meta else meta.get("chunk_index")
            label = f"[S{i}]"
            target = f"{source}#{chunk}" if chunk is not None else source
            citations.append({"id": label, "target": target})
            text = (getattr(d, "page_content", "") or "").strip()
            context_parts.append(f"{label} {target}\n{text}")
        return "\n\n".join(context_parts), citations

    def _build_prompt(self, question: str, context: str) -> str:
        # Only include last 2 history entries — fewer tokens = faster on CPU
        history_text = ""
        if self.history:
            recent = self.history[-2:]
            history_text = "\n".join(f"{r}: {c[:100]}" for r, c in recent)
        if context.strip():
            return (
                f"{_SYSTEM_PROMPT_RAG}\n"
                f"Context:\n{context}\n\n"
                f"Q: {question}\nA:"
            )
        return (
            f"{_SYSTEM_PROMPT_CHAT}\n"
            f"{history_text}\n"
            f"Q: {question}\nA:"
        )

    def _trim_history(self):
        if len(self.history) > MAX_HISTORY:
            self.history = self.history[-MAX_HISTORY:]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _hybrid_retrieve(self, question: str, k: int) -> List:
        """Combine vector search with BM25 keyword search for better recall.

        On low-core CPUs (<=4), runs searches sequentially to avoid thread
        pool overhead. The BM25 search is fast enough in-process.
        """
        # Retrieve a modest pool — keep it small on CPU-bound systems
        retrieval_pool = min(k * 2, 12)
        retriever = self.db.as_retriever(search_kwargs={"k": retrieval_pool})

        t0 = time.time()

        # Vector search
        if hasattr(retriever, "invoke"):
            vector_docs = retriever.invoke(question)
        elif hasattr(retriever, "get_relevant_documents"):
            vector_docs = retriever.get_relevant_documents(question)
        else:
            raise RuntimeError("Retriever does not support known retrieval API")

        # BM25 search (sequential — cheaper than thread pool on 2-core CPU)
        bm25_docs = []
        if self._bm25_data is not None:
            bm25, all_docs = self._bm25_data
            scores = bm25.get_scores(question.lower().split())
            top_idx = np.argsort(scores)[-retrieval_pool:][::-1]
            bm25_docs = [all_docs[i] for i in top_idx if scores[i] > 0]

        if not bm25_docs:
            result = self._rerank(question, vector_docs, k)
            logger.debug(
                "Retrieval (vector-only): %d candidates → %d results in %.2fs",
                len(vector_docs), len(result), time.time() - t0,
            )
            return result

        # Merge, deduplicate by content
        seen = set()
        merged = []
        for d in vector_docs + bm25_docs:
            content = (getattr(d, "page_content", "") or "").strip()
            if content not in seen:
                seen.add(content)
                merged.append(d)

        result = self._rerank(question, merged, k)
        logger.debug(
            "Retrieval (hybrid): %d vector + %d bm25 = %d merged → %d results in %.2fs",
            len(vector_docs), len(bm25_docs), len(merged), len(result), time.time() - t0,
        )
        return result

    def _rerank(self, question: str, docs: List, k: int) -> List:
        if _reranker is None or len(docs) <= 1:
            return docs[:k]
        try:
            pairs = [(question, (getattr(d, "page_content", "") or "").strip()) for d in docs]
            scores = _reranker.predict(pairs)
            ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
            return [doc for _, doc in ranked[:k]]
        except Exception as e:
            logger.debug("Re-ranking failed: %s", e)
            return docs[:k]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, question: str, k: int = None, remember: bool = True) -> Tuple[str, List]:
        if self.db is None:
            raise RuntimeError("Vectorstore not loaded. Run: python scripts/rebuild_vectorstore.py")

        k = k or RETRIEVER_K

        greeting = self._check_greeting(question)
        if greeting:
            if remember:
                with self._history_lock:
                    self.history.append(("user", question))
                    self.history.append(("assistant", greeting))
                    self._trim_history()
            return greeting, []

        if remember:
            with self._history_lock:
                self.history.append(("user", question))

        # Check cache first (only for non-remembered queries or exact repeats)
        cached = self._cache.get(question, k)
        if cached:
            answer, docs = cached
            logger.debug("Cache hit for: %s", question[:60])
            if remember:
                with self._history_lock:
                    self.history.append(("assistant", answer))
                    self._trim_history()
            return answer, docs

        t0 = time.time()
        docs = self._hybrid_retrieve(self._enrich_query(question), k)
        context, _ = self._build_context(docs)
        prompt = self._build_prompt(question, context)

        if not hasattr(self.llm, "invoke"):
            raise RuntimeError("LLM must expose an `invoke(prompt)` method")

        answer = self.llm.invoke(prompt)
        elapsed = time.time() - t0
        logger.info(
            "ask() completed: %.1fs | docs=%d | answer_len=%d | q='%s'",
            elapsed, len(docs), len(answer), question[:60],
        )

        # Cache the result
        self._cache.put(question, k, answer, docs)

        if remember:
            with self._history_lock:
                self.history.append(("assistant", answer))
                self._trim_history()
        return answer, docs

    def ask_stream(self, question: str, k: int = None, remember: bool = True):
        """Like ask(), but yields (token, docs) chunks for streaming UX."""
        if self.db is None:
            raise RuntimeError("Vectorstore not loaded.")

        k = k or RETRIEVER_K

        greeting = self._check_greeting(question)
        if greeting:
            if remember:
                with self._history_lock:
                    self.history.append(("user", question))
                    self.history.append(("assistant", greeting))
                    self._trim_history()
            yield greeting, []
            return

        if remember:
            with self._history_lock:
                self.history.append(("user", question))

        t0 = time.time()
        docs = self._hybrid_retrieve(self._enrich_query(question), k)
        self._last_docs = docs
        context, _ = self._build_context(docs)
        prompt = self._build_prompt(question, context)

        full_answer = []
        for token in self.llm.stream(prompt):
            full_answer.append(str(token))
            yield str(token), docs

        answer_text = "".join(full_answer)
        elapsed = time.time() - t0
        logger.info(
            "ask_stream() completed: %.1fs | docs=%d | answer_len=%d | q='%s'",
            elapsed, len(docs), len(answer_text), question[:60],
        )

        # Cache the full result for future non-streaming calls
        self._cache.put(question, k, answer_text, docs)

        if remember:
            with self._history_lock:
                self.history.append(("assistant", answer_text))
                self._trim_history()

    def suggest_followups(self, question: str, answer: str, num: int = 3) -> List[str]:
        prompt = (
            f"Based on this veterinary Q&A, suggest exactly {num} short follow-up questions "
            "a student would ask next. Return ONLY the questions, one per line, no numbering.\n\n"
            f"Question: {question}\nAnswer: {answer[:300]}\n\nFollow-up questions:"
        )
        try:
            raw = self.llm.invoke(prompt)
            lines = [l.strip().lstrip("0123456789.-) ") for l in raw.strip().split("\n") if l.strip()]
            return [l for l in lines if len(l) > 10 and l.endswith("?")][:num]
        except Exception:
            return []

    def reset_history(self):
        with self._history_lock:
            self.history = []
        self._cache.clear()

    def get_history(self) -> List[tuple]:
        with self._history_lock:
            return list(self.history)

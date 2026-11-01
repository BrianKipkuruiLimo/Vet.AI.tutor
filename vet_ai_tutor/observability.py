"""Logging setup, request IDs, per-stage timings and in-memory metrics.

Standard library only. Each answered question produces one summary line
showing where the time went (retrieval vs. time to first token vs.
generation), and the same numbers feed `metrics`, which /dev/metrics
reports as medians and 95th percentiles over the last requests.
"""
import json
import logging
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Optional

from vet_ai_tutor.config import LOG_FORMAT, LOG_LEVEL

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# Attributes every LogRecord has; anything else was passed via `extra=`.
_STANDARD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime", "request_id"}


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line: time, level, logger, message, request_id and extras."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        entry.update({k: v for k, v in vars(record).items() if k not in _STANDARD_ATTRS})
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str, ensure_ascii=False)


_HANDLER_NAME = "vet-ai-tutor"


def configure_logging(level: str = LOG_LEVEL, fmt: str = LOG_FORMAT, stream=None) -> None:
    """Install one root handler (text or JSON). Safe to call more than once."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler.get_name() == _HANDLER_NAME:
            root.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(JsonFormatter() if fmt == "json" else logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s [%(request_id)s] %(message)s"))
    root.addHandler(handler)
    root.setLevel(level)
    # httpx logs every Ollama call at INFO; the per-question summary covers it.
    logging.getLogger("httpx").setLevel(logging.WARNING)


class StageTimer:
    """Accumulates wall-clock seconds per named stage of one request."""

    def __init__(self):
        self.started = time.perf_counter()
        self.stages: Dict[str, float] = {}
        self.values: Dict[str, object] = {}

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t0)

    def add(self, name: str, seconds: float) -> None:
        self.stages[name] = self.stages.get(name, 0.0) + seconds

    def since_start(self) -> float:
        return time.perf_counter() - self.started

    def summary_line(self) -> str:
        s, v = self.stages, self.values
        parts = [f"done in {v.get('total_seconds', self.since_start()):.1f}s"]
        if v.get("cache_hit"):
            parts.append("cache hit")
        retrieval = [k for k in ("vector_search", "bm25_search", "fusion", "rerank") if k in s]
        if retrieval:
            detail = ", ".join(f"{k.split('_')[0]} {s[k]:.2f}" for k in retrieval)
            parts.append(f"retrieval {sum(s[k] for k in retrieval):.2f}s ({detail})")
        if "first_token" in v:
            reading = f" (prompt processing {s['prompt_processing']:.1f}s)" if "prompt_processing" in s else ""
            parts.append(f"first token {v['first_token']:.1f}s{reading}")
        if "generation" in s:
            rate = v.get("tokens_per_second")
            parts.append(f"generation {s['generation']:.1f}s" + (f", {rate:.1f} tok/s" if rate else ""))
        return " | ".join(parts)


def _percentile(sorted_values: list, q: float) -> float:
    """Nearest-rank percentile of a sorted list (q in 0-100)."""
    if not sorted_values:
        return 0.0
    rank = max(1, round(q / 100 * len(sorted_values)))
    return sorted_values[min(rank, len(sorted_values)) - 1]


class Metrics:
    """Rolling per-stage statistics over the last `window` questions (thread-safe)."""

    def __init__(self, window: int = 500):
        self._lock = threading.Lock()
        self._window = window
        self._stages: Dict[str, deque] = {}
        self.questions = 0
        self.cache_hits = 0

    def observe(self, timer: StageTimer) -> None:
        values = dict(timer.stages)
        for key in ("total_seconds", "first_token", "tokens_per_second"):
            if isinstance(timer.values.get(key), (int, float)):
                values[key] = float(timer.values[key])  # type: ignore[arg-type]
        with self._lock:
            self.questions += 1
            self.cache_hits += bool(timer.values.get("cache_hit"))
            for name, seconds in values.items():
                self._stages.setdefault(name, deque(maxlen=self._window)).append(seconds)

    def summary(self) -> dict:
        with self._lock:
            stages = {name: sorted(values) for name, values in self._stages.items()}
            questions, hits = self.questions, self.cache_hits
        return {
            "questions": questions,
            "cache_hit_rate": round(hits / questions, 3) if questions else None,
            "stages": {
                name: {"count": len(v), "p50": round(_percentile(v, 50), 3),
                       "p95": round(_percentile(v, 95), 3), "max": round(v[-1], 3)}
                for name, v in stages.items()
            },
        }

    def reset(self) -> None:
        with self._lock:
            self._stages.clear()
            self.questions = self.cache_hits = 0


metrics = Metrics()


def record_question(timer: StageTimer, logger: Optional[logging.Logger] = None, **fields) -> None:
    """Finish a question: store its metrics and log one summary line."""
    timer.values.setdefault("total_seconds", timer.since_start())
    metrics.observe(timer)
    (logger or logging.getLogger("vet_ai_tutor.timing")).info(
        "question %s", timer.summary_line(),
        extra={"event": "question", "stages": {k: round(v, 4) for k, v in timer.stages.items()},
               **{k: v for k, v in timer.values.items()}, **fields},
    )

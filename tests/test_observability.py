"""Tests for logging, stage timings and metrics (vet_ai_tutor.observability)."""
import io
import json
import logging

import pytest

from tests.test_core import FakeDB, FakeDoc, FakeLLM, _make_tutor
from vet_ai_tutor import observability as obs


@pytest.fixture(autouse=True)
def fresh_metrics():
    obs.metrics.reset()
    yield
    obs.metrics.reset()


def test_stage_timer_accumulates_and_summarizes():
    timer = obs.StageTimer()
    with timer.stage("vector_search"):
        pass
    timer.add("vector_search", 0.25)
    timer.add("bm25_search", 0.05)
    timer.add("generation", 12.0)
    timer.add("prompt_processing", 8.0)
    timer.values.update(first_token=8.5, tokens_per_second=17.2, total_seconds=20.0)
    assert timer.stages["vector_search"] >= 0.25
    line = timer.summary_line()
    assert line.startswith("done in 20.0s")
    assert "retrieval 0.30s (vector 0.25, bm25 0.05)" in line
    assert "first token 8.5s (prompt processing 8.0s)" in line and "generation 12.0s, 17.2 tok/s" in line


def test_percentiles_and_metrics_summary():
    assert obs._percentile([], 50) == 0.0
    values = sorted(float(i) for i in range(1, 101))
    assert obs._percentile(values, 50) == 50.0 and obs._percentile(values, 95) == 95.0
    m = obs.Metrics(window=3)
    for seconds, hit in ((1.0, False), (2.0, True), (3.0, False), (4.0, False)):
        t = obs.StageTimer()
        t.add("generation", seconds)
        t.values["cache_hit"] = hit
        m.observe(t)
    summary = m.summary()
    assert summary["questions"] == 4 and summary["cache_hit_rate"] == 0.25
    assert summary["stages"]["generation"]["count"] == 3  # rolling window
    assert summary["stages"]["generation"]["max"] == 4.0


def test_json_logs_carry_request_id_and_extras():
    stream = io.StringIO()
    obs.configure_logging(level="INFO", fmt="json", stream=stream)
    token = obs.request_id_var.set("abc12345")
    try:
        logging.getLogger("test.json").info("hello %s", "world", extra={"event": "x", "seconds": 1.5})
    finally:
        obs.request_id_var.reset(token)
        obs.configure_logging(level="WARNING", fmt="text")  # restore a quiet handler
    entry = json.loads(stream.getvalue().strip().splitlines()[-1])
    assert entry["message"] == "hello world" and entry["request_id"] == "abc12345"
    assert entry["event"] == "x" and entry["seconds"] == 1.5 and entry["level"] == "INFO"


def test_configure_logging_is_idempotent():
    obs.configure_logging(fmt="text")
    obs.configure_logging(fmt="text")
    names = [h.get_name() for h in logging.getLogger().handlers]
    assert names.count("vet-ai-tutor") == 1


def test_tutor_records_stage_timings_for_ask_and_stream(caplog):
    t = _make_tutor(llm=FakeLLM("Milk fever is hypocalcaemia [S1]."), db=FakeDB([FakeDoc("passage")]))
    with caplog.at_level(logging.INFO, logger="vet_ai_tutor.core"):
        t.ask("What is milk fever?", k=1, remember=False)
        list(t.ask_stream("What is mastitis in cows?", k=1, remember=False))
        t.ask("What is milk fever?", k=1, remember=False)  # cache hit
    summary = obs.metrics.summary()
    assert summary["questions"] == 3
    assert summary["cache_hit_rate"] == round(1 / 3, 3)
    for stage in ("vector_search", "fusion", "rerank", "prompt", "generation", "prompt_processing", "first_token",
                  "tokens_per_second", "total_seconds"):
        assert stage in summary["stages"], stage
    records = [r for r in caplog.records if getattr(r, "event", "") == "question"]
    assert len(records) == 3
    assert records[1].mode == "stream" and records[1].first_token >= 0
    assert records[2].cache_hit is True and "cache hit" in records[2].getMessage()

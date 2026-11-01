"""HTTP-level tests for vet_ai_tutor.api using a fake Tutor (no Ollama needed)."""
import json

import pytest
from fastapi.testclient import TestClient

from tests.test_core import FakeDoc, FakeLLM, _make_tutor
from vet_ai_tutor import api


@pytest.fixture
def tutor():
    docs = [FakeDoc("Milk fever is hypocalcaemia.", "dairy.pdf", 3)]
    docs[0].metadata.update({"source_tier": "tier_1", "source_title": "Dairy Medicine",
                             "source_url": "https://www.merckvetmanual.com/x"})
    from tests.test_core import FakeDB
    return _make_tutor(llm=FakeLLM("Milk fever is low calcium [S1]."), db=FakeDB(docs))


@pytest.fixture
def client(tutor, monkeypatch):
    # TestClient without a `with` block skips the lifespan, so no Ollama is needed.
    monkeypatch.setitem(api._state, "tutor", tutor)
    monkeypatch.setitem(api._state, "init_error", None)
    api._request_log.clear()
    return TestClient(api.app)


def _parse_sse(text: str) -> list:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_ask_returns_answer_sources_and_citations(client):
    res = client.post("/ask", json={"question": "What is milk fever?", "session_id": "tab-12345678"})
    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == "Milk fever is low calcium [S1]."
    assert body["sources"][0]["source_title"] == "Dairy Medicine"
    assert body["citations"] == [{"id": "[S1]", "target": "dairy.pdf#3"}]
    assert body["session_id"] == "tab-12345678"
    assert res.headers["X-Request-ID"]


def test_ask_stream_sends_sources_then_tokens_then_done(client):
    res = client.post("/ask_stream", json={"question": "What is milk fever?"})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(res.text)
    names = [name for name, _ in events]
    assert names[0] == "sources" and names[-1] == "done"
    assert events[0][1][0]["source"] == "dairy.pdf"
    assert "".join(data for name, data in events if name == "token").strip() == "Milk fever is low calcium [S1]."


def test_ask_stream_reports_errors_as_events(client, tutor):
    def boom(*a, **k):
        raise RuntimeError("secret internal detail")
        yield  # pragma: no cover
    tutor.ask_stream = boom
    events = _parse_sse(client.post("/ask_stream", json={"question": "What is milk fever?"}).text)
    assert [name for name, _ in events] == ["error", "done"]
    assert "secret internal detail" not in events[0][1]["detail"]


def test_sessions_are_separate_and_resettable(client):
    client.post("/ask", json={"question": "What is milk fever?", "session_id": "student-aaaa"})
    client.post("/ask", json={"question": "What is mastitis?", "session_id": "student-bbbb"})
    a = client.get("/history", params={"session_id": "student-aaaa"}).json()
    assert [e["content"] for e in a["history"] if e["role"] == "user"] == ["What is milk fever?"]
    assert client.post("/reset", json={"session_id": "student-aaaa"}).status_code == 200
    assert client.get("/history", params={"session_id": "student-aaaa"}).json()["count"] == 0
    assert client.get("/history", params={"session_id": "student-bbbb"}).json()["count"] == 2


def test_validation_errors_use_the_error_envelope(client):
    res = client.post("/ask", json={"question": "", "session_id": "bad id!"})
    assert res.status_code == 422
    body = res.json()
    assert body["error"] == "validation_error"
    assert "question" in body["detail"] and "session_id" in body["detail"]
    assert body["request_id"]


def test_question_length_is_limited(client):
    res = client.post("/ask", json={"question": "x" * 5000})
    assert res.status_code == 422


def test_unloaded_tutor_returns_503_with_reason(client, monkeypatch):
    monkeypatch.setitem(api._state, "tutor", None)
    monkeypatch.setitem(api._state, "init_error", "Ollama is not running")
    res = client.post("/ask", json={"question": "What is milk fever?"})
    assert res.status_code == 503
    assert res.json() == {"error": "unavailable", "detail": "Ollama is not running",
                          "request_id": res.headers["X-Request-ID"]}
    health = client.get("/health").json()
    assert health["status"] == "degraded" and health["warning"] == "Ollama is not running"


def test_rate_limit(client, monkeypatch):
    monkeypatch.setattr(api, "RATE_LIMIT_PER_MINUTE", 2)
    for _ in range(2):
        assert client.post("/ask", json={"question": "What is milk fever?"}).status_code == 200
    res = client.post("/ask", json={"question": "What is milk fever?"})
    assert res.status_code == 429
    assert res.json()["error"] == "rate_limited"
    assert client.get("/health").status_code == 200  # other endpoints unaffected


def test_unhandled_error_hides_internals(client, tutor):
    def boom(**kwargs):
        raise RuntimeError("database password is hunter2")
    tutor.ask = boom
    res = TestClient(api.app, raise_server_exceptions=False).post(
        "/ask", json={"question": "What is milk fever?"})
    assert res.status_code == 500
    assert "hunter2" not in res.text
    assert res.json()["error"] == "internal_server_error"


def test_health_reports_loaded_index(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["vectorstore_loaded"] is True


def test_admin_routes_require_token(client, monkeypatch):
    monkeypatch.setattr(api, "ADMIN_TOKEN", "")
    assert client.get("/dev/health").status_code == 503
    monkeypatch.setattr(api, "ADMIN_TOKEN", "correct-token")
    assert client.get("/dev/health").status_code == 401
    assert client.get("/dev/health", headers={"X-Admin-Token": "wrong"}).status_code == 401
    assert client.get("/dev/health", headers={"X-Admin-Token": "correct-token"}).status_code == 200


def test_cors_does_not_allow_arbitrary_origins(client):
    res = client.options("/ask", headers={
        "Origin": "https://evil.example", "Access-Control-Request-Method": "POST",
    })
    assert res.headers.get("access-control-allow-origin") != "https://evil.example"


def test_dev_metrics_requires_token_and_reports_stages(client, monkeypatch):
    from vet_ai_tutor.observability import metrics
    metrics.reset()
    monkeypatch.setattr(api, "ADMIN_TOKEN", "correct-token")
    assert client.get("/dev/metrics").status_code == 401
    client.post("/ask", json={"question": "What is milk fever?"})
    body = client.get("/dev/metrics", headers={"X-Admin-Token": "correct-token"}).json()
    assert body["questions"] == 1
    assert {"vector_search", "generation", "total_seconds"} <= set(body["stages"])
    assert body["stages"]["generation"]["p50"] >= 0
    health = client.get("/health").json()
    assert health["performance"]["questions"] == 1 and "total_seconds" in health["performance"]["median"]

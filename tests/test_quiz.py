"""Tests for quiz mode: validation, the bank, review/publish, and the API."""
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vet_ai_tutor import quiz

ROOT = Path(__file__).resolve().parents[1]
PASSAGE = ("Bovine babesiosis (redwater) is caused by Babesia bigemina and Babesia bovis, "
           "transmitted by Rhipicephalus ticks. Signs include fever, haemoglobinuria and anaemia. "
           "Imidocarb dipropionate is used for treatment and short-term prevention.")
GOOD = {
    "question": "Which ticks transmit Babesia bigemina in cattle?",
    "options": ["Rhipicephalus ticks", "Culicoides midges", "Tabanid flies", "Ixodes ticks"],
    "answer_index": 0,
    "explanation": "The passage states babesiosis is transmitted by Rhipicephalus ticks.",
}


def load_script(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def entry(question=GOOD["question"], source="babesiosis.pdf", status="approved", **extra):
    e = quiz.make_entry({**GOOD, "question": question}, PASSAGE, {"source": source, "page": 3})
    e["status"] = status
    e.update(extra)
    return e


# ---------------------------------------------------------------------------
# Validation and parsing
# ---------------------------------------------------------------------------

def test_valid_question_passes():
    assert quiz.validate_question(GOOD, PASSAGE) == []


@pytest.mark.parametrize("change, problem", [
    ({"question": "Ticks"}, "ending with '?'"),
    ({"options": ["a", "b", "c"]}, "exactly 4"),
    ({"options": ["Rhipicephalus ticks"] * 4}, "duplicate"),
    ({"options": ["Rhipicephalus ticks", "Midges", "Flies", "All of the above"]}, "above"),
    ({"answer_index": 4}, "answer_index"),
    ({"answer_index": True}, "answer_index"),
    ({"answer_index": 1}, "not supported by the passage"),  # "Culicoides midges" isn't in the passage
    ({"explanation": ""}, "explanation"),
    # A "wrong" option that the passage states is a second correct answer.
    ({"options": ["Rhipicephalus ticks", "Imidocarb dipropionate treatment", "Tabanid flies", "Ixodes ticks"]},
     "two correct answers"),
])
def test_invalid_questions_are_rejected(change, problem):
    problems = quiz.validate_question({**GOOD, **change}, PASSAGE)
    assert any(problem in p for p in problems), problems


def test_parse_generated_handles_fences_and_garbage():
    assert quiz.parse_generated('```json\n{"question": "x?"}\n```') == {"question": "x?"}
    assert quiz.parse_generated('Sure! {"a": 1} hope that helps') == {"a": 1}
    assert quiz.parse_generated("no json here") is None
    assert quiz.parse_generated("[1, 2]") is None


def test_generate_question_with_fake_llm():
    class FakeLLM:
        def __init__(self, reply):
            self.reply = reply

        def invoke(self, prompt):
            assert "Rhipicephalus" in prompt
            return self.reply

    made, problems = quiz.generate_question(FakeLLM(json.dumps(GOOD)), PASSAGE, {"source": "b.pdf", "slide": 2})
    assert problems == [] and made["status"] == "pending_review" and made["slide"] == 2
    bad, problems = quiz.generate_question(FakeLLM("not json"), PASSAGE, {"source": "b.pdf"})
    assert bad is None and problems == ["model did not return valid JSON"]


# ---------------------------------------------------------------------------
# Bank
# ---------------------------------------------------------------------------

def test_bank_serves_only_approved_questions_without_answers(tmp_path):
    path = tmp_path / quiz.BANK_FILE
    quiz.save_bank([entry(), entry("Which drug treats bovine babesiosis?", status="pending_review"),
                    entry("What colour is the urine in redwater?", source="urine.pdf")], str(path))
    bank = quiz.QuizBank(str(path))
    assert len(bank) == 2
    picked = bank.pick(5, rng=random.Random(1))
    assert len(picked) == 2 and all(set(q) == {"id", "question", "options"} for q in picked)
    assert [q["question"] for q in bank.pick(5, topic="urine")] == ["What colour is the urine in redwater?"]
    assert bank.pick(5, exclude=[q["id"] for q in picked]) == []

    qid = picked[0]["id"]
    right = bank.check(qid, 0)
    assert right["correct"] and right["source"] and right["explanation"]
    assert not bank.check(qid, 2)["correct"]
    assert bank.check("missing", 0) is None
    assert {t["topic"] for t in bank.topics()} == {"babesiosis.pdf", "urine.pdf"}


# ---------------------------------------------------------------------------
# Review, publish and passage selection
# ---------------------------------------------------------------------------

def test_review_decisions_and_publish(tmp_path):
    review = load_script("review_quiz")
    a, b, c = (entry(q, status="pending_review") for q in (
        "Which ticks transmit Babesia bigemina in cattle?",
        "Which drug is used to treat bovine babesiosis?",
        "Which sign is typical of redwater in cattle?"))
    assert review.apply_decision(a, "a") and a["status"] == "approved"
    assert review.apply_decision(b, "3") and b["status"] == "approved" and b["answer_index"] == 2
    assert review.apply_decision(c, "r") and c["status"] == "rejected"
    assert not review.apply_decision(a, "x")

    bank_file = str(tmp_path / quiz.BANK_FILE)
    assert review.publish([a, b, c], bank_file) == 2
    c["status"] = "approved"
    assert review.publish([c], bank_file) == 3   # earlier approvals are kept
    a["status"] = "rejected"
    assert review.publish([a], bank_file) == 2   # a later rejection removes the question


def test_passage_selection_round_robins_sources():
    gen = load_script("generate_quiz")
    long = "x" * 500
    records = [(long + str(i), {"source": src}) for src in ("a.pdf", "b.pdf", "c.pdf") for i in range(6)]
    picked = gen.select_passages(records, per_source=2, sources=[], used=set(), limit=3)
    assert [m["source"] for _, m in picked] == ["a.pdf", "b.pdf", "c.pdf"]
    short = [("too short", {"source": "a.pdf"})]
    assert gen.select_passages(short, 2, [], set(), 0) == []
    used = {gen.passage_key(records[0][0])}
    assert records[0] not in gen.select_passages(records, 6, ["a.pdf"], used, 0)


def test_rebuild_keeps_quiz_bank(tmp_path):
    from langchain_core.documents import Document
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from vet_ai_tutor.ingestion.indexer import build_index, chunk_documents

    target = tmp_path / "index"
    chunks = chunk_documents([Document(page_content="Redwater notes. " * 40, metadata={"source": "a"})])
    build_index(chunks, DeterministicFakeEmbedding(size=8), str(target), str(tmp_path / "b"), progress=lambda _m: None)
    quiz.save_bank([entry()], str(target / quiz.BANK_FILE))
    build_index(chunks, DeterministicFakeEmbedding(size=8), str(target), str(tmp_path / "b"), progress=lambda _m: None)
    assert len(quiz.QuizBank(str(target / quiz.BANK_FILE))) == 1


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    from vet_ai_tutor import api
    path = tmp_path / quiz.BANK_FILE
    quiz.save_bank([entry(), entry("Which drug is used to treat bovine babesiosis?")], str(path))
    monkeypatch.setitem(api._state, "quiz", quiz.QuizBank(str(path)))
    api._request_log.clear()
    return TestClient(api.app)


def test_quiz_endpoints(client):
    body = client.get("/quiz", params={"n": 5}).json()
    assert body["available"] == 2 and len(body["questions"]) == 2
    assert "answer_index" not in body["questions"][0]
    qid = body["questions"][0]["id"]
    result = client.post("/quiz/check", json={"id": qid, "choice": 0}).json()
    assert result["correct"] is True and result["page"] == 3
    assert client.post("/quiz/check", json={"id": "nope", "choice": 0}).status_code == 404
    assert client.post("/quiz/check", json={"id": qid, "choice": 7}).status_code == 422
    assert client.get("/quiz", params={"n": 50}).status_code == 400
    assert client.get("/quiz/topics").json()["total"] == 2

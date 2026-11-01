"""Quiz mode: a reviewed bank of multiple-choice questions grounded in sources.

Generating a question takes 20-30 s on a laptop CPU, so questions are made
ahead of time by developers (scripts/generate_quiz.py), validated here,
approved by a person (scripts/review_quiz.py), and published to
<index>/quiz_bank.json. Living inside the index directory, the bank ships
with signed knowledge patches and offline bundles and survives rebuilds.

At runtime students get questions without answers; a choice is graded by
id on the server, which returns the explanation and the source to cite.
"""
import hashlib
import json
import os
import random
import re
import threading
from typing import Dict, List, Optional

from vet_ai_tutor.answer_checks import content_words
from vet_ai_tutor.config import VECTOR_PATH, project_path

BANK_FILE = "quiz_bank.json"
OPTION_COUNT = 4
# Share of the correct option's content words that must appear in the passage.
MIN_ANSWER_GROUNDING = 0.5
# A wrong option this well supported by the passage is probably a second correct answer.
MAX_DISTRACTOR_GROUNDING = 0.8

GENERATION_PROMPT = """You write multiple-choice questions for veterinary students.
Using ONLY the passage below, write ONE question that tests an important fact from it.

Rules:
- Exactly 4 options. Exactly one is correct and is stated in the passage.
- The 3 wrong options must be plausible but clearly wrong according to the passage.
- No "all of the above" or "none of the above".
- The explanation quotes or closely paraphrases the passage.

Return JSON only:
{{"question": "...?", "options": ["...", "...", "...", "..."], "answer_index": 0, "explanation": "..."}}

Passage (from {source}):
\"\"\"
{passage}
\"\"\"
"""

_BANNED_OPTIONS = re.compile(r"\b(all|none|both) of the above\b", re.IGNORECASE)


def bank_path(index_dir: str = VECTOR_PATH) -> str:
    return os.path.join(project_path(index_dir), BANK_FILE)


def question_id(question: str, source: str) -> str:
    return hashlib.sha256(f"{source}|{question.strip().lower()}".encode()).hexdigest()[:12]


def parse_generated(raw: str) -> Optional[dict]:
    """Extract the JSON object from a model response, or None."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def validate_question(q: dict, passage: str) -> List[str]:
    """Return the reasons a generated question is unusable (empty list = valid)."""
    problems = []
    question = str(q.get("question", "")).strip()
    options = q.get("options")
    if len(question) < 15 or not question.endswith("?"):
        problems.append("question must be a full sentence ending with '?'")
    if not isinstance(options, list) or len(options) != OPTION_COUNT:
        return problems + [f"needs exactly {OPTION_COUNT} options"]
    options = [str(o).strip() for o in options]
    if any(not o for o in options):
        problems.append("empty option")
    if len({o.lower() for o in options}) != OPTION_COUNT:
        problems.append("duplicate options")
    if any(_BANNED_OPTIONS.search(o) for o in options):
        problems.append("'all/none of the above' options are not allowed")
    index = q.get("answer_index")
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < OPTION_COUNT:
        return problems + ["answer_index must be 0-3"]
    answer_words = content_words(options[index])
    passage_words = content_words(passage)
    if answer_words and len(answer_words & passage_words) / len(answer_words) < MIN_ANSWER_GROUNDING:
        problems.append("correct option is not supported by the passage")
    for i, option in enumerate(options):
        words = content_words(option)
        if i != index and len(words) >= 2 and len(words & passage_words) / len(words) >= MAX_DISTRACTOR_GROUNDING:
            problems.append(f"wrong option {i + 1} is also supported by the passage (two correct answers?)")
    if not str(q.get("explanation", "")).strip():
        problems.append("missing explanation")
    return problems


def make_entry(q: dict, passage: str, meta: dict) -> dict:
    """Normalize a validated question into a bank entry with its source."""
    source = meta.get("source") or meta.get("source_file") or "unknown"
    entry = {
        "id": question_id(q["question"], source),
        "question": q["question"].strip(),
        "options": [str(o).strip() for o in q["options"]],
        "answer_index": q["answer_index"],
        "explanation": q["explanation"].strip(),
        "source": source,
        "source_title": meta.get("source_title", ""),
        "source_tier": meta.get("source_tier", ""),
        "passage": passage.strip()[:1200],
        "status": "pending_review",
    }
    for key in ("page", "slide", "species", "topic"):
        if meta.get(key) not in (None, ""):
            entry[key] = meta[key]
    return entry


def generate_question(llm, passage: str, meta: dict) -> tuple:
    """Ask the LLM for one question; return (entry or None, problems)."""
    source = meta.get("source_title") or meta.get("source") or "lecture notes"
    raw = llm.invoke(GENERATION_PROMPT.format(source=source, passage=passage.strip()[:2500]))
    q = parse_generated(raw)
    if q is None:
        return None, ["model did not return valid JSON"]
    problems = validate_question(q, passage)
    return (None, problems) if problems else (make_entry(q, passage, meta), [])


def load_bank(path: Optional[str] = None) -> List[dict]:
    path = path or bank_path()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    return [q for q in data.get("questions", []) if q.get("status") == "approved"]


def save_bank(questions: List[dict], path: Optional[str] = None) -> None:
    path = path or bank_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"format": 1, "questions": questions}, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


class QuizBank:
    """Approved questions, loaded once. Thread-safe for the API's worker threads."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or bank_path()
        self._lock = threading.Lock()
        self._by_id: Dict[str, dict] = {q["id"]: q for q in load_bank(self.path)}

    def __len__(self) -> int:
        return len(self._by_id)

    def topics(self) -> List[dict]:
        counts: Dict[str, int] = {}
        for q in self._by_id.values():
            title = q.get("source_title") or q["source"]
            counts[title] = counts.get(title, 0) + 1
        return [{"topic": t, "questions": n} for t, n in sorted(counts.items())]

    def pick(self, n: int, topic: str = "", exclude: Optional[List[str]] = None,
             rng: Optional[random.Random] = None) -> List[dict]:
        """Random questions (without answers), optionally filtered by a topic phrase."""
        exclude_set = set(exclude or [])
        needle = topic.strip().lower()
        with self._lock:
            pool = [q for q in self._by_id.values() if q["id"] not in exclude_set and (
                not needle or needle in " ".join(
                    [q["question"], q.get("source_title", ""), q["source"], q.get("topic", "")]).lower())]
        chosen = (rng or random).sample(pool, min(n, len(pool)))
        return [{"id": q["id"], "question": q["question"], "options": q["options"]} for q in chosen]

    def check(self, qid: str, choice: int) -> Optional[dict]:
        q = self._by_id.get(qid)
        if q is None:
            return None
        return {
            "id": qid,
            "correct": choice == q["answer_index"],
            "answer_index": q["answer_index"],
            "explanation": q["explanation"],
            "source": q["source"],
            "source_title": q.get("source_title", ""),
            "page": q.get("page", ""),
            "slide": q.get("slide", ""),
        }

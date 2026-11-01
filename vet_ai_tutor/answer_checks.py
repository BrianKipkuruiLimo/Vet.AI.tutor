"""Fast, deterministic answer-quality checks (no LLM judge).

Cheap enough to run on every benchmark, even on a laptop CPU:

- citations:   does the answer cite passages, and only ones that were retrieved?
- not_found:   did the tutor decline ("Not found in my knowledge base.")?
- key_terms:   share of the reference answer's content words in the answer
- grounding:   share of the answer's content words found in the retrieved
               passages (or the question). Low grounding means the model
               added terms its sources don't contain, a cheap hallucination signal.

For LLM-judged metrics (RAGAS faithfulness etc.) see scripts/benchmark.py --ragas.
"""
import re
from typing import Iterable, List

NOT_FOUND = "not found in my knowledge base"

_CITATION_RE = re.compile(r"\[S(\d+)\]")
_WORD_RE = re.compile(r"[a-z][a-z0-9-]+")
_STOPWORDS = {
    "about", "after", "also", "although", "among", "and", "any", "are", "because", "been",
    "before", "being", "between", "both", "but", "can", "cause", "caused", "causes", "could",
    "does", "due", "during", "each", "either", "etc", "for", "from", "has", "have", "having",
    "how", "however", "include", "includes", "including", "into", "its", "may", "more", "most",
    "much", "must", "not", "often", "only", "other", "over", "should", "some", "such", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "through",
    "thus", "under", "until", "usually", "very", "was", "were", "what", "when", "where", "which",
    "while", "who", "will", "with", "within", "without", "would", "you", "your",
}


def content_words(text: str) -> set:
    """Lowercase content words (3+ letters, no stopwords, no citation markers)."""
    text = _CITATION_RE.sub(" ", (text or "").lower())
    return {w for w in _WORD_RE.findall(text) if len(w) > 2 and w not in _STOPWORDS}


def is_not_found(answer: str) -> bool:
    return NOT_FOUND in (answer or "").lower()


def cited_passages(answer: str) -> List[int]:
    return sorted({int(n) for n in _CITATION_RE.findall(answer or "")})


def check_answer(question: str, answer: str, expected: str, passages: Iterable[str]) -> dict:
    """Run all checks for one answer; `passages` are the retrieved texts in [S#] order."""
    passages = list(passages)
    cited = cited_passages(answer)
    invalid = [n for n in cited if not 1 <= n <= len(passages)]
    answer_words = content_words(answer)
    expected_words = content_words(expected)
    support = content_words(question).union(*(content_words(p) for p in passages)) if passages else set()
    not_found = is_not_found(answer)
    return {
        "not_found": not_found,
        "has_citation": bool(cited),
        "cited": cited,
        "invalid_citations": invalid,
        "key_term_recall": round(len(answer_words & expected_words) / len(expected_words), 4)
        if expected_words else None,
        # A declined answer has nothing to ground.
        "grounding": round(len(answer_words & support) / len(answer_words), 4)
        if answer_words and not not_found else None,
        "unsupported_terms": sorted(answer_words - support)[:15] if not not_found else [],
    }


def summarize_checks(checks: List[dict]) -> dict:
    """Aggregate check_answer() results into benchmark summary numbers."""
    n = len(checks)
    answered = [c for c in checks if not c["not_found"]]

    def mean(values):
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 4) if values else None

    return {
        "questions": n,
        "not_found_rate": round(sum(c["not_found"] for c in checks) / n, 4) if n else None,
        "citation_rate": round(sum(c["has_citation"] for c in answered) / len(answered), 4) if answered else None,
        "answers_with_invalid_citations": sum(1 for c in checks if c["invalid_citations"]),
        "mean_key_term_recall": mean(c["key_term_recall"] for c in checks),
        "mean_grounding": mean(c["grounding"] for c in answered),
    }

"""Tests for vet_ai_tutor.core.Tutor.

These tests use lightweight fakes so they run without Ollama, ChromaDB, or
HuggingFace models installed.
"""
import pytest


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeDoc:
    def __init__(self, content="found content", source="fake", chunk=0):
        self.page_content = content
        self.metadata = {"source": source, "chunk": chunk}


class FakeRetriever:
    def __init__(self, docs=None):
        self._docs = docs or [FakeDoc()]

    def get_relevant_documents(self, q):
        return self._docs


class FakeDB:
    def __init__(self, docs=None):
        self._retriever = FakeRetriever(docs)

    def as_retriever(self, search_kwargs=None):
        return self._retriever


class FakeLLM:
    """Fake LLM that records the prompt and returns a canned answer."""
    def __init__(self, answer="LLM answer"):
        self._answer = answer
        self.last_prompt = None
        self.call_count = 0

    def invoke(self, prompt):
        self.last_prompt = prompt
        self.call_count += 1
        return self._answer

    def stream(self, prompt):
        self.last_prompt = prompt
        self.call_count += 1
        for word in self._answer.split():
            yield word + " "


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tutor(**overrides):
    """Create a Tutor instance without calling __init__ (avoids heavy deps)."""
    from vet_ai_tutor.core import Tutor, _QueryCache
    import threading

    t = object.__new__(Tutor)
    t.history = []
    t._history_lock = threading.Lock()
    t.llm = overrides.get("llm", FakeLLM())
    t.db = overrides.get("db", FakeDB())
    t._bm25_data = overrides.get("bm25_data", None)
    t._cache = overrides.get("cache", _QueryCache())
    t._doc_count = overrides.get("doc_count", 1)
    t._backend = overrides.get("backend", "faiss")
    return t


# ---------------------------------------------------------------------------
# Tests: ask()
# ---------------------------------------------------------------------------

class TestAsk:
    def test_basic_ask_returns_answer_and_docs(self):
        t = _make_tutor()
        answer, docs = t.ask("What is milk fever?", k=1, remember=False)
        assert answer == "LLM answer"
        assert isinstance(docs, list)
        assert docs[0].page_content == "found content"

    def test_ask_with_remember_stores_history(self):
        t = _make_tutor()
        t.ask("What is milk fever?", k=1, remember=True)
        assert len(t.history) == 2
        assert t.history[0] == ("user", "What is milk fever?")
        assert t.history[1][0] == "assistant"

    def test_ask_without_remember_does_not_store_history(self):
        t = _make_tutor()
        t.ask("What is milk fever?", k=1, remember=False)
        assert len(t.history) == 0

    def test_ask_raises_when_no_db(self):
        t = _make_tutor(db=None)
        with pytest.raises(RuntimeError, match="Vectorstore not loaded"):
            t.ask("test")


# ---------------------------------------------------------------------------
# Tests: greeting handling
# ---------------------------------------------------------------------------

class TestGreeting:
    @pytest.mark.parametrize("greeting", [
        "Hi", "hello", "Hey", "good morning", "Good Afternoon", "GOOD EVENING",
    ])
    def test_greetings_are_handled_locally(self, greeting):
        llm = FakeLLM()
        t = _make_tutor(llm=llm)
        answer, docs = t.ask(greeting, k=1)
        assert "Vet AI Tutor" in answer
        assert docs == []
        # LLM should NOT have been called for a greeting
        assert llm.last_prompt is None

    def test_non_greeting_calls_llm(self):
        llm = FakeLLM()
        t = _make_tutor(llm=llm)
        t.ask("What is mastitis?", k=1)
        assert llm.last_prompt is not None


# ---------------------------------------------------------------------------
# Tests: query enrichment
# ---------------------------------------------------------------------------

class TestQueryEnrichment:
    def test_short_followup_is_enriched(self):
        t = _make_tutor()
        # Simulate history state after current question is already appended
        # (ask() appends user question before calling _enrich_query)
        t.history = [
            ("user", "What is milk fever in dairy cattle?"),
            ("assistant", "Milk fever is..."),
            ("user", "How is it treated?"),
        ]
        enriched = t._enrich_query("How is it treated?")
        assert "milk fever" in enriched.lower()
        assert "How is it treated?" in enriched

    def test_long_question_is_not_enriched(self):
        t = _make_tutor()
        t.history = [("user", "prior question")]
        long_q = "What are the detailed clinical signs and pathophysiology of milk fever?"
        enriched = t._enrich_query(long_q)
        assert enriched == long_q

    def test_no_history_returns_original(self):
        t = _make_tutor()
        enriched = t._enrich_query("How is it treated?")
        assert enriched == "How is it treated?"


# ---------------------------------------------------------------------------
# Tests: history management
# ---------------------------------------------------------------------------

class TestHistoryManagement:
    def test_reset_history_clears(self):
        t = _make_tutor()
        t.history = [("user", "q1"), ("assistant", "a1")]
        t.reset_history()
        assert t.history == []

    def test_get_history_returns_copy(self):
        t = _make_tutor()
        t.history = [("user", "q1")]
        h = t.get_history()
        h.append(("user", "q2"))
        assert len(t.history) == 1  # original unmodified

    def test_history_is_trimmed(self):
        t = _make_tutor()
        # Fill history beyond MAX_HISTORY
        for i in range(30):
            t.history.append(("user", f"q{i}"))
        t._trim_history()
        from vet_ai_tutor.config import MAX_HISTORY
        assert len(t.history) == MAX_HISTORY


# ---------------------------------------------------------------------------
# Tests: ask_stream()
# ---------------------------------------------------------------------------

class TestAskStream:
    def test_stream_yields_tokens(self):
        t = _make_tutor()
        tokens = []
        for token, docs in t.ask_stream("What is milk fever?", k=1, remember=False):
            tokens.append(token)
        assert len(tokens) > 0
        full = "".join(tokens)
        assert "LLM" in full

    def test_stream_greeting(self):
        t = _make_tutor()
        tokens = list(t.ask_stream("Hello", k=1))
        assert len(tokens) == 1
        assert "Vet AI Tutor" in tokens[0][0]


# ---------------------------------------------------------------------------
# Tests: context building
# ---------------------------------------------------------------------------

class TestBuildContext:
    def test_builds_labeled_context(self):
        t = _make_tutor()
        docs = [FakeDoc("Milk fever content", "dairy.pdf", 5)]
        context, citations = t._build_context(docs)
        assert "[S1]" in context
        assert "dairy.pdf#5" in context
        assert "Milk fever content" in context
        assert citations[0]["id"] == "[S1]"

    def test_empty_docs_produce_empty_context(self):
        t = _make_tutor()
        context, citations = t._build_context([])
        assert context == ""
        assert citations == []


# ---------------------------------------------------------------------------
# Tests: LRU query cache
# ---------------------------------------------------------------------------

class TestQueryCache:
    def test_cache_hit_avoids_llm_call(self):
        """Second identical question should be served from cache."""
        llm = FakeLLM()
        t = _make_tutor(llm=llm)

        # First call — cache miss, LLM is called
        ans1, _ = t.ask("What is milk fever?", k=1, remember=False)
        assert llm.call_count == 1

        # Second call — cache hit, LLM should NOT be called again
        ans2, _ = t.ask("What is milk fever?", k=1, remember=False)
        assert llm.call_count == 1  # still 1
        assert ans1 == ans2

    def test_cache_is_case_insensitive(self):
        """Cache key normalizes case."""
        llm = FakeLLM()
        t = _make_tutor(llm=llm)

        t.ask("What is Milk Fever?", k=1, remember=False)
        assert llm.call_count == 1

        t.ask("what is milk fever?", k=1, remember=False)
        assert llm.call_count == 1  # cache hit

    def test_reset_clears_cache(self):
        """Resetting history also clears the query cache."""
        llm = FakeLLM()
        t = _make_tutor(llm=llm)

        t.ask("What is milk fever?", k=1, remember=False)
        assert llm.call_count == 1

        t.reset_history()

        # After reset, cache is cleared — LLM should be called again
        t.ask("What is milk fever?", k=1, remember=False)
        assert llm.call_count == 2

    def test_cache_stats(self):
        """Cache tracks hits and misses."""
        t = _make_tutor()
        t.ask("What is milk fever?", k=1, remember=False)
        t.ask("What is milk fever?", k=1, remember=False)
        t.ask("What is mastitis?", k=1, remember=False)

        stats = t.cache_stats
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["size"] == 2


# ---------------------------------------------------------------------------
# Tests: properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_doc_count(self):
        t = _make_tutor(doc_count=42)
        assert t.doc_count == 42

    def test_backend(self):
        t = _make_tutor(backend="chroma")
        assert t.backend == "chroma"

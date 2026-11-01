"""Tests for embeddings, the LLM adapter and the evaluation/drafting scripts."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class RecordingEmbeddings:
    def __init__(self):
        self.documents, self.queries = [], []

    def embed_documents(self, texts):
        self.documents.extend(texts)
        return [[1.0] for _ in texts]

    def embed_query(self, text):
        self.queries.append(text)
        return [1.0]


def test_nomic_prefixes_are_applied():
    from vet_ai_tutor.embeddings import PrefixedEmbeddings, prefixes_for
    query_prefix, document_prefix = prefixes_for("nomic-embed-text:latest", "auto")
    assert (query_prefix, document_prefix) == ("search_query: ", "search_document: ")
    base = RecordingEmbeddings()
    emb = PrefixedEmbeddings(base, query_prefix, document_prefix)
    emb.embed_documents(["Milk fever notes"])
    emb.embed_query("what is milk fever")
    assert base.documents == ["search_document: Milk fever notes"]
    assert base.queries == ["search_query: what is milk fever"]


def test_prefixes_off_and_unknown_models():
    from vet_ai_tutor.embeddings import prefixes_for
    assert prefixes_for("nomic-embed-text", "off") == ("", "")
    assert prefixes_for("bge-m3", "auto") == ("", "")
    assert prefixes_for("qwen3-embedding:0.6b", "auto")[0].startswith("Instruct:")


def test_index_profile_mismatch_is_reported(tmp_path):
    from vet_ai_tutor.embeddings import check_index_profile, embedding_profile, write_index_info
    assert "no index_info.json" in check_index_profile(str(tmp_path))
    write_index_info(str(tmp_path))
    assert check_index_profile(str(tmp_path)) is None
    old = {**embedding_profile(), "document_prefix": ""}
    write_index_info(str(tmp_path), old)
    assert "Rebuild the index" in check_index_profile(str(tmp_path), embedding_profile("nomic-embed-text", "auto"))


# ---------------------------------------------------------------------------
# LLM adapter
# ---------------------------------------------------------------------------

class StreamingLLM:
    def __init__(self, chunks):
        self.chunks = chunks

    def invoke(self, prompt):
        return "".join(self.chunks)

    def stream(self, prompt):
        yield from self.chunks


def test_invoke_strips_think_blocks():
    from vet_ai_tutor.llm_adapter import OllamaAdapter
    llm = OllamaAdapter(StreamingLLM(["<think>\nreasoning\n</think>\n\nMilk fever [S1]."]))
    assert llm.invoke("q") == "Milk fever [S1]."


def test_stream_hides_think_block_split_across_chunks():
    from vet_ai_tutor.llm_adapter import OllamaAdapter
    chunks = ["<th", "ink>", "let me ", "think</th", "ink>\n", "Milk ", "fever."]
    assert "".join(OllamaAdapter(StreamingLLM(chunks)).stream("q")) == "Milk fever."


def test_stream_passes_normal_answers_through():
    from vet_ai_tutor.llm_adapter import OllamaAdapter
    chunks = ["Milk", " fever", " is <b>low</b> calcium."]
    assert list(OllamaAdapter(StreamingLLM(chunks)).stream("q")) == chunks


def test_build_llm_disables_qwen3_thinking_and_sets_timeout():
    from vet_ai_tutor.llm_adapter import build_llm
    llm = build_llm("qwen3:1.7b")._llm
    assert llm.reasoning is False
    assert llm.client_kwargs["timeout"] > 0
    assert build_llm("llama3.2:3b")._llm.reasoning is None


# ---------------------------------------------------------------------------
# Evaluation and metadata drafting scripts
# ---------------------------------------------------------------------------

class Doc:
    def __init__(self, text, source="", page=None):
        self.page_content = text
        self.metadata = {"source": source, "page": page}


def test_evaluation_matches_by_source_and_page():
    ev = load_script("evaluate_retrieval")
    item = {"question": "q", "source": "raw_notes/dairy.pdf", "page": 4}
    docs = [Doc("x", "other.pdf", 4), Doc("x", "dairy.pdf", 2), Doc("x", "dairy.pdf", 4)]
    assert ev.first_relevant_rank(docs, item, 0.5) == 3


def test_evaluation_answer_overlap_fallback():
    ev = load_script("evaluate_retrieval")
    item = {"question": "q", "answer": "Heat, redness, swelling, pain and loss of function."}
    docs = [Doc("Cardiac output depends on heart rate."),
            Doc("Inflammation signs: heat, redness, swelling, pain, loss of function.")]
    assert ev.first_relevant_rank(docs, item, 0.5) == 2


def test_evaluation_summary_metrics():
    ev = load_script("evaluate_retrieval")
    summary = ev.summarize([1, 2, None, 6], k=5)
    assert summary["hits"] == 2
    assert summary["recall_at_k"] == 0.5
    assert abs(summary["mrr"] - (1 + 0.5 + 1 / 6) / 4) < 1e-9


def test_draft_metadata_needs_review(tmp_path):
    drafter = load_script("draft_source_metadata")
    path = tmp_path / "26. VMED 344 Diseases of Foals-Dr. Kirui G 2021.txt"
    path.write_text("notes", encoding="utf-8")
    entry = drafter.draft_entry(path, path.name, "BVM Year 4", "")
    assert entry["_needs_review"] is True
    assert entry["source_tier"] == ""
    assert entry["publication_year"] == "2021"
    assert entry["author"].startswith("Dr. Kirui")
    assert entry["journal"] == "BVM Year 4"
    assert entry["source_url"].startswith("file://raw_notes/")
    json.dumps(entry)  # must be serializable


# ---------------------------------------------------------------------------
# Answer checks
# ---------------------------------------------------------------------------

def test_answer_checks_citations_and_grounding():
    from vet_ai_tutor.answer_checks import check_answer
    passages = ["Milk fever is hypocalcaemia in dairy cows around calving.",
                "Treat with slow intravenous calcium borogluconate."]
    good = check_answer("What is milk fever?",
                        "Milk fever is hypocalcaemia around calving [S1]; treat with calcium borogluconate [S2].",
                        "Hypocalcaemia around calving.", passages)
    assert good["has_citation"] and good["cited"] == [1, 2] and not good["invalid_citations"]
    assert good["grounding"] == 1.0
    assert good["key_term_recall"] == 1.0  # hypocalcaemia, around, calving all present

    bad = check_answer("What is milk fever?",
                       "It is caused by Brucella abortus infection [S4].", "Hypocalcaemia.", passages)
    assert bad["invalid_citations"] == [4]
    assert bad["grounding"] < 0.5 and "brucella" in bad["unsupported_terms"]
    assert bad["key_term_recall"] == 0.0


def test_answer_checks_not_found_and_summary():
    from vet_ai_tutor.answer_checks import check_answer, summarize_checks
    declined = check_answer("q", "Not found in my knowledge base.", "Something specific.", ["passage"])
    assert declined["not_found"] and declined["grounding"] is None
    answered = check_answer("q", "Passage text here [S1].", "passage text", ["passage text here"])
    summary = summarize_checks([declined, answered])
    assert summary["not_found_rate"] == 0.5
    assert summary["citation_rate"] == 1.0
    assert summary["mean_grounding"] == 1.0


# ---------------------------------------------------------------------------
# QA source review tool
# ---------------------------------------------------------------------------

def test_review_choice_records_source_and_page():
    review = load_script("review_qa_sources")
    candidates = [Doc("a", "dairy.pdf", 4), Doc("b", "slides.pptx")]
    candidates[1].metadata = {"source": "slides.pptx", "slide": 7}
    item = {"id": "q1", "question": "q", "answer": "a"}
    assert review.apply_choice(item, "2", candidates)
    assert item == {"id": "q1", "question": "q", "answer": "a", "source": "slides.pptx",
                    "slide": 7, "source_reviewed": True}
    assert review.apply_choice(item, "1", candidates)  # re-pick replaces slide with page
    assert item["source"] == "dairy.pdf" and item["page"] == 4 and "slide" not in item
    assert not review.apply_choice({}, "9", candidates)
    manual = {}
    assert review.apply_choice(manual, "m", candidates, "Pharmacology notes.pdf#12")
    assert manual["source"] == "Pharmacology notes.pdf" and manual["page"] == 12
    missed = {}
    assert review.apply_choice(missed, "n", candidates) and missed["retrieval_missed"]


def test_review_session_saves_progress(tmp_path, monkeypatch):
    review = load_script("review_qa_sources")
    path = tmp_path / "qa.json"
    items = [{"id": "q1", "question": "Q1", "answer": "A1"},
             {"id": "q2", "question": "Q2", "answer": "A2", "source": "x.pdf"},
             {"id": "q3", "question": "Q3", "answer": "A3"}]
    review.save_items(path, items)
    answers = iter(["1", "q"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    done = review.review(items, lambda q, n: [Doc("text", "notes.pdf", 2)], path, 3, review_all=False)
    saved = json.loads(path.read_text())
    assert done == 1
    assert saved[0]["source"] == "notes.pdf" and saved[0]["page"] == 2
    assert "source" not in saved[2]  # quit before q3; q2 already had a source

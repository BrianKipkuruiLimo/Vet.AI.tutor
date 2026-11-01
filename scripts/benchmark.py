#!/usr/bin/env python3
"""Benchmark the Vet AI Tutor — Full Pipeline Evaluation.

Tests answer correctness, response times, and reports system specs.
Requires Ollama to be running and the vectorstore to be built.

Usage:
    python scripts/benchmark.py
"""
import json
import os
import sys
import time
import platform
import subprocess

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vet_ai_tutor.config import VECTOR_PATH, EMBEDDING_MODEL, LLM_MODEL, RETRIEVER_K, VECTOR_BACKEND


# ---------------------------------------------------------------------------
# 1. System Specs
# ---------------------------------------------------------------------------
def get_system_specs():
    """Collect system hardware and software information."""
    specs = {
        "os": f"{platform.system()} {platform.release()}",
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "processor": platform.processor() or "unknown",
    }

    # CPU info
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
        model_names = [line.split(":")[1].strip() for line in cpuinfo.split("\n") if "model name" in line]
        if model_names:
            specs["cpu_model"] = model_names[0]
            specs["cpu_cores"] = len(model_names)
    except Exception:
        specs["cpu_model"] = specs["processor"]
        try:
            specs["cpu_cores"] = os.cpu_count()
        except Exception:
            specs["cpu_cores"] = "unknown"

    # RAM
    try:
        with open("/proc/meminfo") as f:
            meminfo = f.read()
        for line in meminfo.split("\n"):
            if "MemTotal" in line:
                kb = int(line.split()[1])
                specs["ram_total_gb"] = round(kb / 1024 / 1024, 1)
                break
    except Exception:
        specs["ram_total_gb"] = "unknown"

    # GPU (nvidia)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            specs["gpu"] = result.stdout.strip()
        else:
            specs["gpu"] = "none detected"
    except Exception:
        specs["gpu"] = "none detected (nvidia-smi not found)"

    # Pipeline config
    specs["embedding_model"] = EMBEDDING_MODEL
    specs["llm_model"] = LLM_MODEL
    specs["vector_backend"] = VECTOR_BACKEND
    specs["vector_path"] = VECTOR_PATH
    specs["retriever_k"] = RETRIEVER_K

    return specs


# ---------------------------------------------------------------------------
# 2. Semantic Similarity (for scoring)
# ---------------------------------------------------------------------------
def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    import numpy as np
    a, b = np.array(a), np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# 3. Keyword overlap scoring
# ---------------------------------------------------------------------------
def keyword_overlap(expected: str, actual: str) -> float:
    """Simple token-level F1 between expected and actual answers."""
    expected_tokens = set(expected.lower().split())
    actual_tokens = set(actual.lower().split())

    # Remove very common words
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "to",
                 "and", "or", "for", "that", "this", "it", "its", "with", "by",
                 "on", "at", "as", "from", "be", "can", "has", "have", "had"}
    expected_tokens -= stopwords
    actual_tokens -= stopwords

    if not expected_tokens:
        return 0.0

    overlap = expected_tokens & actual_tokens
    if not overlap:
        return 0.0

    precision = len(overlap) / len(actual_tokens) if actual_tokens else 0
    recall = len(overlap) / len(expected_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# 4. Grade an answer
# ---------------------------------------------------------------------------
def grade_answer(expected: str, actual: str, embeddings=None) -> dict:
    """Grade an answer using keyword overlap + semantic similarity."""
    kw_score = keyword_overlap(expected, actual)

    sem_score = 0.0
    if embeddings:
        try:
            exp_emb = embeddings.embed_query(expected)
            act_emb = embeddings.embed_query(actual)
            sem_score = cosine_similarity(exp_emb, act_emb)
        except Exception:
            pass

    # Combined score (weighted)
    combined = 0.4 * kw_score + 0.6 * sem_score

    # Classification
    if combined >= 0.70:
        status = "CORRECT"
    elif combined >= 0.50:
        status = "PARTIAL"
    else:
        status = "INCORRECT"

    return {
        "keyword_f1": round(kw_score, 4),
        "semantic_sim": round(sem_score, 4),
        "combined_score": round(combined, 4),
        "status": status,
    }


# ---------------------------------------------------------------------------
# 5. Main benchmark
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark answer quality and speed.")
    parser.add_argument("--qa", default=os.path.join("data", "eval", "qa_pairs.json"))
    parser.add_argument("--limit", type=int, help="only the first N questions (quick check)")
    parser.add_argument("--ragas", action="store_true",
                        help="also score with RAGAS using a local judge model (slow; needs `uv sync --group eval`)")
    parser.add_argument("--judge-model", default="qwen3:4b", help="Ollama model used as the RAGAS judge")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print("=" * 70)
    print("  🐄 VET AI TUTOR — FULL PIPELINE BENCHMARK")
    print("=" * 70)
    print()

    # ── System specs ──
    print("📊 Collecting system specs...")
    specs = get_system_specs()
    print(f"   OS:        {specs['os']}")
    print(f"   CPU:       {specs.get('cpu_model', 'unknown')} ({specs.get('cpu_cores', '?')} cores)")
    print(f"   RAM:       {specs.get('ram_total_gb', '?')} GB")
    print(f"   GPU:       {specs.get('gpu', 'none')}")
    print(f"   Python:    {specs['python_version']}")
    print(f"   LLM:       {specs['llm_model']}")
    print(f"   Embeddings: {specs['embedding_model']}")
    print(f"   VectorDB:  {specs['vector_backend']} @ {specs['vector_path']}")
    print(f"   Retriever K: {specs['retriever_k']}")
    print()

    # ── Load Tutor ──
    print("🔧 Initializing Tutor...")
    init_start = time.time()

    from vet_ai_tutor.core import Tutor
    from vet_ai_tutor.llm_adapter import build_llm

    adapter = build_llm()  # same settings as the app
    tutor = Tutor(llm=adapter, vector_path=VECTOR_PATH)

    init_time = time.time() - init_start
    print(f"   Tutor initialized in {init_time:.1f}s")

    if tutor.db is None:
        print("❌ Vectorstore not loaded! Run rebuild_vectorstore.py first.")
        sys.exit(1)

    # Get embeddings for semantic scoring
    embeddings = tutor.embeddings

    # ── Load Q&A pairs ──
    qa_path = args.qa
    if not os.path.exists(qa_path):
        print(f"❌ Q&A file not found: {qa_path}")
        sys.exit(1)

    with open(qa_path, encoding="utf-8") as f:
        qa_pairs = json.load(f)[: args.limit]

    total = len(qa_pairs)
    print(f"\n📝 Running {total} questions through the full RAG pipeline...")
    print("   (Each question → retrieve → generate → grade)")
    print("-" * 70)

    from vet_ai_tutor.answer_checks import check_answer, summarize_checks

    # ── Run evaluation ──
    results = []
    checks = []
    ragas_rows = []
    times = []
    correct = 0
    partial = 0
    incorrect = 0

    for i, qa in enumerate(qa_pairs, 1):
        qid = qa.get("id", f"q{i}")
        question = qa["question"]
        expected = qa["answer"]

        print(f"\n[{i}/{total}] {qid}: {question[:70]}...")

        # Time the full ask() call
        t0 = time.time()
        try:
            answer, docs = tutor.ask(question, remember=False)
        except Exception as e:
            answer = f"ERROR: {e}"
            docs = []
        elapsed = time.time() - t0
        times.append(elapsed)

        # Grade
        grade = grade_answer(expected, answer, embeddings)
        passages = [getattr(d, "page_content", "") for d in docs]
        check = check_answer(question, answer, expected, passages)
        checks.append(check)
        ragas_rows.append({"user_input": question, "response": answer,
                           "retrieved_contexts": passages, "reference": expected})

        # Count
        if grade["status"] == "CORRECT":
            correct += 1
            icon = "✅"
        elif grade["status"] == "PARTIAL":
            partial += 1
            icon = "🟡"
        else:
            incorrect += 1
            icon = "❌"

        print(f"   {icon} {grade['status']}  score={grade['combined_score']:.2f}  "
              f"kw_f1={grade['keyword_f1']:.2f}  sem={grade['semantic_sim']:.2f}  "
              f"time={elapsed:.1f}s  sources={len(docs)}")
        flags = []
        if check["not_found"]:
            flags.append("declined")
        elif not check["has_citation"]:
            flags.append("no citation")
        if check["invalid_citations"]:
            flags.append(f"invalid citations {check['invalid_citations']}")
        if check["grounding"] is not None and check["grounding"] < 0.6:
            flags.append(f"low grounding {check['grounding']:.2f} {check['unsupported_terms'][:5]}")
        if flags:
            print(f"   ⚠️  {'; '.join(flags)}")

        result = {
            "id": qid,
            "question": question,
            "expected": expected,
            "actual_answer": answer[:500],
            "num_sources": len(docs),
            "time_seconds": round(elapsed, 2),
            **grade,
            "checks": check,
        }
        results.append(result)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  📊 BENCHMARK RESULTS")
    print("=" * 70)

    total_time = sum(times)
    avg_time = total_time / len(times) if times else 0
    min_time = min(times) if times else 0
    max_time = max(times) if times else 0

    accuracy = (correct / total * 100) if total else 0
    partial_pct = (partial / total * 100) if total else 0
    incorrect_pct = (incorrect / total * 100) if total else 0

    avg_combined = sum(r["combined_score"] for r in results) / len(results) if results else 0
    avg_semantic = sum(r["semantic_sim"] for r in results) / len(results) if results else 0
    avg_kw = sum(r["keyword_f1"] for r in results) / len(results) if results else 0
    check_summary = summarize_checks(checks)

    def pct(value):
        return "n/a" if value is None else f"{value:.0%}"

    print(f"""
  Accuracy:
    ✅ Correct:    {correct}/{total}  ({accuracy:.1f}%)
    🟡 Partial:    {partial}/{total}  ({partial_pct:.1f}%)
    ❌ Incorrect:  {incorrect}/{total}  ({incorrect_pct:.1f}%)

  Scores (averages):
    Combined:      {avg_combined:.3f}
    Semantic Sim:  {avg_semantic:.3f}
    Keyword F1:    {avg_kw:.3f}

  Answer checks (no LLM judge):
    Declined ("not found"): {pct(check_summary['not_found_rate'])}
    Cited a source:         {pct(check_summary['citation_rate'])} of answered questions
    Invalid citations:      {check_summary['answers_with_invalid_citations']} answer(s)
    Key-term recall:        {pct(check_summary['mean_key_term_recall'])}
    Grounding in sources:   {pct(check_summary['mean_grounding'])}

  Timing:
    Total time:    {total_time:.1f}s
    Avg per Q:     {avg_time:.1f}s
    Fastest:       {min_time:.1f}s
    Slowest:       {max_time:.1f}s

  System:
    CPU:           {specs.get('cpu_model', 'unknown')}
    RAM:           {specs.get('ram_total_gb', '?')} GB
    GPU:           {specs.get('gpu', 'none')}
    LLM:           {specs['llm_model']}
    Embeddings:    {specs['embedding_model']}
    Vector DB:     {specs['vector_backend']} ({RETRIEVER_K} docs retrieved)
""")

    # ── Save results ──
    timestamp = int(time.time())
    out_path = os.path.join("data", "eval", f"benchmark_{VECTOR_BACKEND}_{timestamp}.json")
    report = {
        "timestamp": timestamp,
        "system_specs": specs,
        "summary": {
            "total_questions": total,
            "correct": correct,
            "partial": partial,
            "incorrect": incorrect,
            "accuracy_pct": round(accuracy, 1),
            "partial_pct": round(partial_pct, 1),
            "incorrect_pct": round(incorrect_pct, 1),
            "avg_combined_score": round(avg_combined, 4),
            "avg_semantic_sim": round(avg_semantic, 4),
            "avg_keyword_f1": round(avg_kw, 4),
            "total_time_seconds": round(total_time, 1),
            "avg_time_per_question": round(avg_time, 2),
            "min_time_seconds": round(min_time, 2),
            "max_time_seconds": round(max_time, 2),
            "init_time_seconds": round(init_time, 1),
        },
        "answer_checks": check_summary,
        "results": results,
    }
    if args.ragas:
        report["ragas"] = run_ragas(ragas_rows, args.judge_model)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  📁 Full report saved: {out_path}")
    print("=" * 70)


def run_ragas(rows: list, judge_model: str) -> dict:
    """Score answers with RAGAS using a local Ollama judge. Slow on CPU; run overnight."""
    try:
        from vet_ai_tutor.ragas_eval import evaluate_rows
    except ImportError as exc:
        print(f"  RAGAS is not installed ({exc}). Install with: uv sync --group eval")
        return {"error": "ragas not installed"}
    print(f"\n🧑‍⚖️ RAGAS with judge {judge_model} on {len(rows)} answers (this is slow on CPU)...")
    scores = evaluate_rows(rows, judge_model)
    for name, value in scores["summary"].items():
        print(f"   {name}: {value:.3f}" if value is not None else f"   {name}: n/a")
    return scores


if __name__ == "__main__":
    main()

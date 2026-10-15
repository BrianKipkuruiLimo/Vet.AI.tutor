"""Evaluate retrieval recall against QA pairs using the current pipeline.

Uses the same Ollama + ChromaDB backend as the production app.
"""
import json
import argparse
import os
import sys
from pathlib import Path
import time
import numpy as np

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vet_ai_tutor.config import (
    VECTOR_PATH, EMBEDDING_MODEL, EMBEDDING_BACKEND,
    VECTOR_BACKEND, RETRIEVER_K,
)


def normalize(text: str) -> str:
    return (text or "").strip().lower()


def cosine_sim(a, b) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def load_embeddings():
    """Load embeddings using the same backend as the app."""
    if EMBEDDING_BACKEND == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
            emb = OllamaEmbeddings(model=EMBEDDING_MODEL)
            emb.embed_query("test")  # smoke test
            print(f"Using OllamaEmbeddings: {EMBEDDING_MODEL}")
            return emb
        except Exception as e:
            print(f"Ollama failed: {e}. Falling back to HuggingFace.")

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        hf_model = EMBEDDING_MODEL if EMBEDDING_BACKEND == "huggingface" else "sentence-transformers/all-MiniLM-L6-v2"
        emb = HuggingFaceEmbeddings(model_name=hf_model)
        print(f"Using HuggingFaceEmbeddings: {hf_model}")
        return emb
    except Exception as e:
        print(f"HuggingFace also failed: {e}")
        raise SystemExit(1)


def load_vectorstore(embeddings):
    """Load vectorstore using the configured backend."""
    if VECTOR_BACKEND == "chroma":
        try:
            from langchain_chroma import Chroma
            db = Chroma(persist_directory=VECTOR_PATH, embedding_function=embeddings)
            count = db._collection.count()
            print(f"ChromaDB loaded: {count} documents from {VECTOR_PATH}")
            return db
        except Exception as e:
            print(f"ChromaDB failed: {e}. Trying FAISS fallback.")

    try:
        from langchain_community.vectorstores import FAISS
        db = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)
        print(f"FAISS loaded from {VECTOR_PATH}")
        return db
    except Exception as e:
        print(f"FAISS also failed: {e}")
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description="Evaluate retrieval recall against QA pairs")
    p.add_argument("--qa", default="data/eval/qa_pairs.json", help="QA JSON file")
    p.add_argument("--k", type=int, default=RETRIEVER_K, help="retrieval k (top-k)")
    p.add_argument("--sim-threshold", type=float, default=0.6, help="low similarity threshold")
    args = p.parse_args()

    qa_path = Path(args.qa)
    if not qa_path.exists():
        print(f"QA file not found: {qa_path}")
        return

    qas = json.loads(qa_path.read_text(encoding="utf8"))
    print(f"Loaded {len(qas)} QA pairs from {qa_path}")

    print(f"\nLoading {EMBEDDING_BACKEND}/{EMBEDDING_MODEL} + {VECTOR_BACKEND}...")
    embeddings = load_embeddings()
    db = load_vectorstore(embeddings)
    retriever = db.as_retriever(search_kwargs={"k": args.k})

    total = len(qas)
    recall_hits = 0
    avg_best_sim = 0.0
    results = []

    print(f"\nEvaluating retrieval with k={args.k}...")
    print("-" * 70)

    for item in qas:
        q = item.get("question", "")
        expected = item.get("answer", "")

        if hasattr(retriever, "invoke"):
            docs = retriever.invoke(q)
        elif hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(q)
        else:
            docs = db.similarity_search(q, k=args.k)

        found = False
        best_sim = 0.0

        # Exact substring check first
        for d in docs:
            content = getattr(d, "page_content", str(d))
            if normalize(expected) in normalize(content):
                found = True
                best_sim = 1.0
                break

        # Cosine similarity if no exact match
        if not found:
            try:
                exp_emb = embeddings.embed_query(expected)
                contents = [getattr(d, "page_content", str(d)) for d in docs]
                if contents:
                    doc_embs = embeddings.embed_documents(contents)
                    for emb in doc_embs:
                        s = cosine_sim(exp_emb, emb)
                        if s > best_sim:
                            best_sim = s
            except Exception as e:
                print(f"  Embedding error for {item.get('id')}: {e}")

        # Dynamic thresholds
        low_thresh = args.sim_threshold
        high_thresh = min(0.95, low_thresh + 0.2)

        exp_len = len(normalize(expected))
        if exp_len > 80:
            high_thresh = max(0.75, high_thresh - 0.05)
        elif exp_len < 20:
            low_thresh = max(low_thresh, 0.75)
            high_thresh = max(high_thresh, 0.9)

        if found or best_sim >= high_thresh:
            status = "FOUND"
            final_found = True
        elif best_sim >= low_thresh:
            status = "PARTIAL"
            final_found = False
        else:
            status = "MISS"
            final_found = False

        recall_hits += 1 if final_found else 0
        avg_best_sim += best_sim

        print(f"[{item.get('id')}] k={args.k} status={status} sim={best_sim:.3f} -- {q[:70]}")

        results.append({
            "id": item.get("id"),
            "question": q,
            "expected": expected,
            "found": final_found,
            "status": status,
            "best_sim": best_sim,
        })

    avg_best_sim = avg_best_sim / total if total else 0.0
    recall = recall_hits / total if total else 0.0

    print(f"\n{'=' * 70}")
    print(f"RETRIEVAL SUMMARY ({EMBEDDING_BACKEND}/{EMBEDDING_MODEL} + {VECTOR_BACKEND})")
    print(f"{'=' * 70}")
    print(f"  Total QA pairs:    {total}")
    print(f"  Recall@{args.k}:         {recall_hits}/{total} = {recall:.1%}")
    print(f"  Avg best sim:      {avg_best_sim:.3f}")
    print(f"  FOUND:             {sum(1 for r in results if r['status'] == 'FOUND')}")
    print(f"  PARTIAL:           {sum(1 for r in results if r['status'] == 'PARTIAL')}")
    print(f"  MISS:              {sum(1 for r in results if r['status'] == 'MISS')}")

    out_dir = Path("data/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_k{args.k}_{int(time.time())}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf8")
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()

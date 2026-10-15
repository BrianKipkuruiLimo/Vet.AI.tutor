"""Quick test: follow-up retrieval quality."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_ollama import OllamaLLM
from vet_ai_tutor.core import Tutor
from vet_ai_tutor.llm_adapter import OllamaAdapter
from vet_ai_tutor.config import VECTOR_PATH, LLM_MODEL, RETRIEVER_K

llm = OllamaLLM(model=LLM_MODEL, num_predict=512)
adapter = OllamaAdapter(llm)
tutor = Tutor(llm=adapter, vector_path=VECTOR_PATH)

# First question
print("Q1: What is milk fever in dairy cattle?")
t0 = time.time()
ans1, docs1 = tutor.ask("What is milk fever in dairy cattle?")
t1 = time.time()
print(f"  Time: {t1-t0:.1f}s | Docs: {len(docs1)}")
for i, d in enumerate(docs1, 1):
    src = (d.metadata or {}).get("source", "?")
    print(f"  [{i}] {src}")

# Follow-up
print("\nQ2: How is it treated?")
t2 = time.time()
ans2, docs2 = tutor.ask("How is it treated?")
t3 = time.time()
print(f"  Time: {t3-t2:.1f}s | Docs: {len(docs2)}")
for i, d in enumerate(docs2, 1):
    src = (d.metadata or {}).get("source", "?")
    print(f"  [{i}] {src}")

print(f"\nFollow-up answer:\n{ans2[:400]}")

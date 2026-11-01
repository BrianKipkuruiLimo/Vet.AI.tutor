"""Optional RAGAS scoring with a local Ollama judge (developer tool, slow on CPU).

Install with `uv sync --group eval`. Metrics:
- faithfulness:   share of the answer's claims supported by the retrieved passages
- context_recall: share of the reference answer covered by the retrieved passages

A small judge model is unreliable, so use a larger one than the tutor
(default qwen3:4b) and treat scores as trends between runs, not absolutes.
"""
import math
import sys
import types
from typing import Any, List


def _install_vertexai_shim() -> None:
    """ragas 0.4.3 imports langchain_community.chat_models.vertexai, which
    langchain-community 0.4 removed. ragas only uses the class for isinstance
    checks, so a placeholder module is enough. Remove once ragas fixes this."""
    name = "langchain_community.chat_models.vertexai"
    try:
        __import__(name)
    except ImportError:
        shim = types.ModuleType(name)
        shim.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[attr-defined]
        sys.modules[name] = shim


def evaluate_rows(rows: List[dict], judge_model: str = "qwen3:4b") -> dict:
    """Score rows of {user_input, response, retrieved_contexts, reference}."""
    _install_vertexai_shim()
    from langchain_ollama import ChatOllama
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import context_recall, faithfulness

    from vet_ai_tutor.llm_adapter import supports_thinking

    judge_kwargs: dict[str, Any] = {"model": judge_model, "temperature": 0, "num_ctx": 4096, "num_predict": 1024}
    if supports_thinking(judge_model):
        judge_kwargs["reasoning"] = False
    judge = LangchainLLMWrapper(ChatOllama(**judge_kwargs))
    # Declined or empty answers have no claims to judge.
    usable = [r for r in rows if r["response"].strip() and r["retrieved_contexts"]]
    if not usable:
        return {"judge_model": judge_model, "summary": {}, "rows": []}
    result: Any = evaluate(  # an EvaluationResult unless return_executor=True
        EvaluationDataset.from_list(usable),
        metrics=[faithfulness, context_recall],
        llm=judge,
        # One request at a time: a CPU Ollama server just queues parallel calls.
        run_config=RunConfig(max_workers=1, timeout=600, max_retries=2),
        show_progress=True,
    )
    scores = [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in s.items()}
              for s in result.scores]

    def mean(name):
        values = [s[name] for s in scores if s.get(name) is not None]
        return sum(values) / len(values) if values else None

    return {
        "judge_model": judge_model,
        "summary": {name: mean(name) for name in ("faithfulness", "context_recall")},
        "rows": [{"question": r["user_input"], **s} for r, s in zip(usable, scores, strict=True)],
    }

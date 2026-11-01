import logging
import re
from typing import Any

from vet_ai_tutor.config import (
    LLM_MODEL, LLM_NUM_CTX, LLM_NUM_PREDICT, LLM_REASONING,
    LLM_TEMPERATURE, LLM_TIMEOUT, OLLAMA_KEEP_ALIVE,
)

logger = logging.getLogger(__name__)

# Model families that emit <think> blocks and accept Ollama's `think` option.
_THINKING_FAMILIES = ("qwen3", "deepseek-r1", "magistral")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def supports_thinking(model: str) -> bool:
    return model.lower().startswith(_THINKING_FAMILIES)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from a model response."""
    return _THINK_BLOCK_RE.sub("", text)


class OllamaAdapter:
    """Adapter that normalizes different Ollama LLM wrappers to an `invoke(prompt)` API."""
    def __init__(self, llm):
        self._llm = llm

    def invoke(self, prompt: str) -> str:
        # prefer .invoke if present (existing code uses this), else try calling
        if hasattr(self._llm, "invoke"):
            return strip_thinking(str(self._llm.invoke(prompt)))
        if callable(self._llm):
            return strip_thinking(str(self._llm(prompt)))
        # last resort: try .generate
        if hasattr(self._llm, "generate"):
            return strip_thinking(str(self._llm.generate(prompt)))
        raise RuntimeError("Provided LLM does not expose a callable interface")

    def stream(self, prompt: str):
        """Yield tokens one at a time for streaming UX, hiding any <think> block."""
        if not hasattr(self._llm, "stream"):
            # fallback: yield the full response as one chunk
            yield self.invoke(prompt)
            return
        pending = ""
        thinking = None  # None = not yet known whether the answer opens with <think>
        for chunk in self._llm.stream(prompt):
            chunk = str(chunk)
            if thinking is False:
                yield chunk
                continue
            pending += chunk
            if thinking is None:
                stripped = pending.lstrip()
                if "<think>".startswith(stripped) and len(stripped) < len("<think>"):
                    continue  # could still become "<think>"
                thinking = stripped.startswith("<think>")
                if not thinking:
                    yield pending
                    pending = ""
                    continue
            end = pending.find("</think>")
            if end != -1:
                thinking = False
                rest = pending[end + len("</think>"):].lstrip()
                pending = ""
                if rest:
                    yield rest
        if thinking is None and pending:
            yield pending

    def warm_up(self) -> None:
        """Load the model into Ollama's memory without generating any tokens."""
        model = getattr(self._llm, "model", None)
        if not model:
            return
        try:
            import ollama
            ollama.generate(model=model, prompt="", keep_alive=getattr(self._llm, "keep_alive", None))
        except Exception as e:
            logger.info("Model warm-up skipped: %s", e)


def build_llm(model: str = LLM_MODEL, **overrides) -> OllamaAdapter:
    """Create the configured Ollama LLM wrapped in an OllamaAdapter."""
    from langchain_ollama import OllamaLLM

    params: dict[str, Any] = {
        "model": model,
        "num_ctx": LLM_NUM_CTX,
        "num_predict": LLM_NUM_PREDICT,
        "temperature": LLM_TEMPERATURE,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        # OllamaLLM silently ignores a top-level `timeout`; it belongs to the HTTP client.
        "client_kwargs": {"timeout": LLM_TIMEOUT},
    }
    if supports_thinking(model):
        params["reasoning"] = LLM_REASONING
    params.update(overrides)
    return OllamaAdapter(OllamaLLM(**params))

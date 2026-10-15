class OllamaAdapter:
    """Adapter that normalizes different Ollama LLM wrappers to an `invoke(prompt)` API."""
    def __init__(self, llm):
        self._llm = llm

    def invoke(self, prompt: str) -> str:
        # prefer .invoke if present (existing code uses this), else try calling
        if hasattr(self._llm, "invoke"):
            return self._llm.invoke(prompt)
        if callable(self._llm):
            return self._llm(prompt)
        # last resort: try .generate or .__call__
        if hasattr(self._llm, "generate"):
            return self._llm.generate(prompt)
        raise RuntimeError("Provided LLM does not expose a callable interface")

    def stream(self, prompt: str):
        """Yield tokens one at a time for streaming UX."""
        if hasattr(self._llm, "stream"):
            for chunk in self._llm.stream(prompt):
                yield chunk
        else:
            # fallback: yield the full response as one chunk
            yield self.invoke(prompt)



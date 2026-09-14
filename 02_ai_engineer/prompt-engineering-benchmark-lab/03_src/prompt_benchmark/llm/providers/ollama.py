from __future__ import annotations

import time
from typing import Any

from prompt_benchmark.llm.base import BaseLLMClient
from prompt_benchmark.llm.schemas import LLMResponse, SUPPORT_TICKET_JSON_SCHEMA
from prompt_benchmark.prompts.base import PromptPayload

class OllamaClient(BaseLLMClient):
    """Local LLM adapter. Supports temperature, top_p and top_k through Ollama options."""

    provider = "ollama"

    def __init__(
        self,
        model: str = "llama3.2:3b",
        host: str = "http://localhost:11434",
        max_output_tokens: int = 64,
        temperature: float | None = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = 42,
    ) -> None:
        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc
        self._client = ollama.Client(host=host)
        self.model = model
        self.host = host
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed

    def classify(self, payload: PromptPayload) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if payload.instructions:
            messages.append({"role": "system", "content": payload.instructions})
        messages.append({"role": "user", "content": payload.input_text})

        options: dict[str, Any] = {"num_predict": self.max_output_tokens}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.top_p is not None:
            options["top_p"] = self.top_p
        if self.top_k is not None:
            options["top_k"] = self.top_k
        if self.seed is not None:
            options["seed"] = self.seed

        request: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False, "options": options}
        if payload.structured_output:
            request["format"] = SUPPORT_TICKET_JSON_SCHEMA

        started = time.perf_counter()
        try:
            response = self._call_with_retry(lambda: self._client.chat(**request), "Ollama chat request")
            latency = time.perf_counter() - started
            message = getattr(response, "message", None)
            raw_output = getattr(message, "content", "") or ""
            input_tokens = int(getattr(response, "prompt_eval_count", 0) or 0)
            output_tokens = int(getattr(response, "eval_count", 0) or 0)
            return LLMResponse(raw_output, input_tokens, output_tokens, input_tokens + output_tokens, latency, self.model, self.provider)
        except Exception as exc:
            return self._error_response(started, exc)

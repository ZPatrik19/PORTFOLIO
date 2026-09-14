from __future__ import annotations

import os
import time
from typing import Any

from prompt_benchmark.llm.base import BaseLLMClient
from prompt_benchmark.llm.schemas import LLMResponse, SUPPORT_TICKET_JSON_SCHEMA
from prompt_benchmark.prompts.base import PromptPayload

class OpenRouterClient(BaseLLMClient):
    """OpenRouter OpenAI-compatible chat-completions adapter.

    The default ``openrouter/free`` router chooses from currently available free
    models. Exact model capabilities can vary, so structured-output requests are
    attempted only when the selected route/model supports them.
    """

    provider = "openrouter"

    def __init__(
        self,
        model: str = "openrouter/free",
        max_output_tokens: int = 64,
        temperature: float | None = 0.0,
        top_p: float | None = None,
        seed: int | None = 42,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set. Add a free OpenRouter key to .env or the UI.")
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
                "X-Title": os.getenv("OPENROUTER_APP_NAME", "Prompt Engineering Benchmark Lab"),
            },
        )
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = None
        self.seed = seed

    @staticmethod
    def _response_format() -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "support_ticket_classification",
                "strict": True,
                "schema": SUPPORT_TICKET_JSON_SCHEMA,
            },
        }

    def classify(self, payload: PromptPayload) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if payload.instructions:
            messages.append({"role": "system", "content": payload.instructions})
        messages.append({"role": "user", "content": payload.input_text})
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_output_tokens,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.top_p is not None:
            request["top_p"] = self.top_p
        if self.seed is not None:
            request["seed"] = self.seed
        if payload.structured_output:
            request["response_format"] = self._response_format()

        started = time.perf_counter()
        try:
            response = self._call_with_retry(lambda: self._client.chat.completions.create(**request), "OpenRouter chat request")
            latency = time.perf_counter() - started
            raw_output = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or (input_tokens + output_tokens))
            return LLMResponse(raw_output, input_tokens, output_tokens, total_tokens, latency, self.model, self.provider)
        except Exception as exc:
            return self._error_response(started, exc)

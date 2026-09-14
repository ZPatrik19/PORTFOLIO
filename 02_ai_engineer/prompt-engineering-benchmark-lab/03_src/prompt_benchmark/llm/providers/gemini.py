from __future__ import annotations

import os
import time
from typing import Any

from prompt_benchmark.llm.base import BaseLLMClient
from prompt_benchmark.llm.schemas import LLMResponse, SUPPORT_TICKET_JSON_SCHEMA
from prompt_benchmark.prompts.base import PromptPayload

class GeminiClient(BaseLLMClient):
    """Google Gemini Interactions API adapter. Sampling overrides are optional experiments; Gemini 3.x should normally use model defaults."""

    provider = "gemini"

    def __init__(
        self,
        model: str,
        max_output_tokens: int = 64,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = 42,
        thinking_level: str | None = "minimal",
    ) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Add a Google AI Studio key to .env.")
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.thinking_level = thinking_level

    def classify(self, payload: PromptPayload) -> LLMResponse:
        # The google-genai SDK maps these generation-config fields to Gemini generation controls.
        generation_config: dict[str, Any] = {"max_output_tokens": self.max_output_tokens}
        if self.temperature is not None:
            generation_config["temperature"] = self.temperature
        if self.top_p is not None:
            generation_config["top_p"] = self.top_p
        if self.top_k is not None:
            generation_config["top_k"] = self.top_k
        if self.seed is not None:
            generation_config["seed"] = self.seed
        effective_thinking_level = payload.reasoning_effort or self.thinking_level
        if effective_thinking_level:
            generation_config["thinking_level"] = effective_thinking_level

        request: dict[str, Any] = {
            "model": self.model,
            "input": payload.input_text,
            "generation_config": generation_config,
        }
        if payload.instructions:
            request["system_instruction"] = payload.instructions
        if payload.structured_output:
            request["response_format"] = {
                "type": "text",
                "mime_type": "application/json",
                "schema": SUPPORT_TICKET_JSON_SCHEMA,
            }

        started = time.perf_counter()
        try:
            interaction = self._call_with_retry(lambda: self._client.interactions.create(**request), "Gemini interaction request")
            latency = time.perf_counter() - started
            usage = getattr(interaction, "usage", None)
            input_tokens = int(getattr(usage, "total_input_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "total_output_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or (input_tokens + output_tokens))
            return LLMResponse(
                raw_output=getattr(interaction, "output_text", "") or "",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_seconds=latency,
                model=self.model,
                provider=self.provider,
            )
        except Exception as exc:
            return self._error_response(started, exc)

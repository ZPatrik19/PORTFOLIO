from __future__ import annotations

import os
import time
from typing import Any

from prompt_benchmark.llm.base import BaseLLMClient
from prompt_benchmark.llm.schemas import LLMResponse, SUPPORT_TICKET_JSON_SCHEMA
from prompt_benchmark.prompts.base import PromptPayload

class OpenAIResponsesClient(BaseLLMClient):
    """OpenAI Responses API adapter. Optional and may incur API charges."""

    provider = "openai"

    def __init__(
        self,
        model: str,
        max_output_tokens: int = 64,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = None
        self.seed = seed

    @staticmethod
    def _schema_format() -> dict[str, Any]:
        return {
            "format": {
                "type": "json_schema",
                "name": "support_ticket_classification",
                "strict": True,
                "schema": SUPPORT_TICKET_JSON_SCHEMA,
            }
        }

    def classify(self, payload: PromptPayload) -> LLMResponse:
        request: dict[str, Any] = {
            "model": self.model,
            "input": payload.input_text,
            "max_output_tokens": self.max_output_tokens,
        }
        if payload.instructions:
            request["instructions"] = payload.instructions
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.top_p is not None:
            request["top_p"] = self.top_p
        if payload.structured_output:
            request["text"] = self._schema_format()
        if payload.reasoning_effort:
            request["reasoning"] = {"effort": payload.reasoning_effort}

        started = time.perf_counter()
        try:
            response = self._call_with_retry(lambda: self._client.responses.create(**request), "OpenAI Responses request")
            latency = time.perf_counter() - started
            usage = getattr(response, "usage", None)
            input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or (input_tokens + output_tokens))
            return LLMResponse(
                raw_output=response.output_text or "",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_seconds=latency,
                model=self.model,
                provider=self.provider,
            )
        except Exception as exc:
            return self._error_response(started, exc)

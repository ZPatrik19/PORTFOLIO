"""Shared provider interface and resilience policy."""
from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar

from prompt_benchmark.llm.schemas import LLMResponse
from prompt_benchmark.prompts.base import PromptPayload

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


def is_retryable_provider_error(exc: BaseException) -> bool:
    """Return whether a provider failure is likely transient.

    Authentication, permission and malformed-request failures are deliberately
    excluded from retries. Network timeouts, rate limits and 5xx-like failures
    are considered transient.
    """
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    non_retryable = (
        "auth",
        "permission",
        "invalid api key",
        "unauthorized",
        "forbidden",
        "bad request",
        "400",
        "404",
    )
    if any(token in name or token in message for token in non_retryable):
        return False
    retryable = (
        "timeout",
        "connection",
        "temporar",
        "rate limit",
        "ratelimit",
        "429",
        "500",
        "502",
        "503",
        "504",
        "service unavailable",
        "resource exhausted",
    )
    return any(token in name or token in message for token in retryable)


class BaseLLMClient(ABC):
    """Provider-neutral interface used by benchmark, UI and CLI code."""

    model: str
    provider: str
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_attempts: int = 3

    @abstractmethod
    def classify(self, payload: PromptPayload) -> LLMResponse:
        """Classify/generate from one fully rendered prompt payload."""
        raise NotImplementedError

    def generation_settings(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k}

    def _call_with_retry(self, operation: Callable[[], T], operation_name: str = "provider request") -> T:
        """Execute an external request with bounded exponential backoff.

        The implementation intentionally uses the standard library so the core
        resilience layer remains available even in minimal/offline environments.
        """
        attempts = max(1, int(getattr(self, "max_attempts", 3)))
        last_error: BaseException | None = None
        for attempt in range(1, attempts + 1):
            try:
                return operation()
            except Exception as exc:  # provider SDKs expose different exception hierarchies
                last_error = exc
                if attempt >= attempts or not is_retryable_provider_error(exc):
                    raise
                delay = min(8.0, 0.5 * (2 ** (attempt - 1))) + random.uniform(0.0, 0.2)
                LOGGER.warning(
                    "%s failed (attempt %s/%s): %s. Retrying in %.2fs",
                    operation_name,
                    attempt,
                    attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)
        raise RuntimeError(f"{operation_name} failed") from last_error

    def _error_response(self, started: float, exc: Exception) -> LLMResponse:
        LOGGER.error("%s request failed: %s: %s", self.provider, type(exc).__name__, exc)
        return LLMResponse(
            raw_output="",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            latency_seconds=time.perf_counter() - started,
            model=self.model,
            provider=self.provider,
            error=f"{type(exc).__name__}: {exc}",
        )

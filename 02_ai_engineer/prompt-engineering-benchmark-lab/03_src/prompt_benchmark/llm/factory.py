from __future__ import annotations

import os
from typing import Any

from prompt_benchmark.llm.client import (
    BaseLLMClient,
    GeminiClient,
    GroqClient,
    MockLLMClient,
    OllamaClient,
    OpenAIResponsesClient,
    OpenRouterClient,
)

SUPPORTED_PROVIDERS = ("mock", "ollama", "groq", "gemini", "openrouter", "openai")

PROVIDER_CAPABILITIES: dict[str, dict[str, bool]] = {
    "mock": {"temperature": True, "top_p": True, "top_k": True, "structured_output": True, "reasoning_effort": False},
    "ollama": {"temperature": True, "top_p": True, "top_k": True, "structured_output": True, "reasoning_effort": False},
    "groq": {"temperature": True, "top_p": True, "top_k": False, "structured_output": True, "reasoning_effort": True},
    "gemini": {"temperature": True, "top_p": True, "top_k": True, "structured_output": True, "reasoning_effort": True},
    "openrouter": {"temperature": True, "top_p": True, "top_k": False, "structured_output": True, "reasoning_effort": False},
    "openai": {"temperature": True, "top_p": True, "top_k": False, "structured_output": True, "reasoning_effort": True},
}


def _optional_float(value: Any, fallback: float | None) -> float | None:
    if value is None or value == "":
        return fallback
    return float(value)


def _optional_int(value: Any, fallback: int | None) -> int | None:
    if value is None or value == "":
        return fallback
    return int(value)


def create_llm_client(
    provider: str,
    config: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> BaseLLMClient:
    """Build a provider client. `overrides` is used by decoding-parameter experiments."""
    provider = provider.lower().strip()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider '{provider}'. Choose from {SUPPORTED_PROVIDERS}.")

    overrides = overrides or {}
    max_output_tokens = int(overrides.get("max_output_tokens", config.get("max_output_tokens", 64)))
    random_seed = int(overrides.get("seed", config.get("random_seed", 42)))

    def ov(name: str, env_name: str, default: Any) -> Any:
        if name in overrides:
            return overrides[name]
        return os.getenv(env_name) if os.getenv(env_name) not in (None, "") else default

    if provider == "mock":
        return MockLLMClient(
            temperature=_optional_float(ov("temperature", "MOCK_TEMPERATURE", 0.0), 0.0),
            top_p=_optional_float(ov("top_p", "MOCK_TOP_P", 1.0), 1.0),
            top_k=_optional_int(ov("top_k", "MOCK_TOP_K", 40), 40),
        )

    if provider == "ollama":
        return OllamaClient(
            model=str(ov("model", "OLLAMA_MODEL", "llama3.2:3b")),
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            max_output_tokens=max_output_tokens,
            temperature=_optional_float(ov("temperature", "OLLAMA_TEMPERATURE", 0.0), 0.0),
            top_p=_optional_float(ov("top_p", "OLLAMA_TOP_P", None), None),
            top_k=_optional_int(ov("top_k", "OLLAMA_TOP_K", None), None),
            seed=random_seed,
        )

    if provider == "groq":
        if overrides.get("top_k") is not None:
            raise ValueError("Groq benchmark adapter does not support top_k. Use top_p or temperature instead.")
        return GroqClient(
            model=str(ov("model", "GROQ_MODEL", "openai/gpt-oss-20b")),
            max_output_tokens=max_output_tokens,
            temperature=_optional_float(ov("temperature", "GROQ_TEMPERATURE", 0.0), 0.0),
            top_p=_optional_float(ov("top_p", "GROQ_TOP_P", None), None),
            seed=random_seed,
        )

    if provider == "gemini":
        return GeminiClient(
            model=str(ov("model", "GEMINI_MODEL", "gemini-3.5-flash-lite")),
            max_output_tokens=max_output_tokens,
            temperature=_optional_float(ov("temperature", "GEMINI_TEMPERATURE", None), None),
            top_p=_optional_float(ov("top_p", "GEMINI_TOP_P", None), None),
            top_k=_optional_int(ov("top_k", "GEMINI_TOP_K", None), None),
            seed=random_seed,
            thinking_level=os.getenv("GEMINI_THINKING_LEVEL", "minimal") or None,
        )

    if provider == "openrouter":
        if overrides.get("top_k") is not None:
            raise ValueError("OpenRouter adapter does not expose a portable top_k control. Use top_p or temperature instead.")
        return OpenRouterClient(
            model=str(ov("model", "OPENROUTER_MODEL", "openrouter/free")),
            max_output_tokens=max_output_tokens,
            temperature=_optional_float(ov("temperature", "OPENROUTER_TEMPERATURE", 0.0), 0.0),
            top_p=_optional_float(ov("top_p", "OPENROUTER_TOP_P", None), None),
            seed=random_seed,
        )

    if overrides.get("top_k") is not None:
        raise ValueError("OpenAI Responses adapter does not expose top_k. Use top_p or temperature instead.")
    openai_model = str(ov("model", "OPENAI_MODEL", config.get("model") or "")).strip()
    if not openai_model:
        raise RuntimeError("OPENAI_MODEL is not set. Put a currently available Responses API model ID in .env or pass --model.")
    return OpenAIResponsesClient(
        model=openai_model,
        max_output_tokens=max_output_tokens,
        temperature=_optional_float(ov("temperature", "OPENAI_TEMPERATURE", config.get("temperature")), config.get("temperature")),
        top_p=_optional_float(ov("top_p", "OPENAI_TOP_P", None), None),
        seed=random_seed,
    )

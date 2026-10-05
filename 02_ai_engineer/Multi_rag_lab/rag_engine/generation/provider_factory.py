from __future__ import annotations

from rag_engine.generation.provider_dummy import DummyLLMProvider
from rag_engine.generation.provider_ollama import OllamaProvider


def create_llm_provider(
    provider: str,
    *,
    base_url: str,
    model_name: str,
    connect_timeout: float = 10.0,
    read_timeout: float = 600.0,
    max_retries: int = 1,
    num_predict: int = 256,
    temperature: float = 0.15,
    keep_alive: str = "5m",
):
    if provider.lower() == "dummy":
        return DummyLLMProvider()
    if provider.lower() == "ollama":
        return OllamaProvider(
            base_url,
            model_name,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            max_retries=max_retries,
            num_predict=num_predict,
            temperature=temperature,
            keep_alive=keep_alive,
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")

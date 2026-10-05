from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

from rag_engine.models import RuntimeConfig


ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    runtime: RuntimeConfig = RuntimeConfig()
    llm_provider: str = "ollama"
    ollama_profile: str = "balanced"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "rag-qwen-balanced"
    ollama_connect_timeout_seconds: float = 10.0
    ollama_read_timeout_seconds: float = 600.0
    ollama_max_retries: int = 1
    ollama_num_predict: int = 384
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    multilingual_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    e5_embedding_model: str = "intfloat/multilingual-e5-small"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_context_tokens: int = 1800


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env", override=False)
    models = _load_yaml(ROOT / "config" / "models.yaml")
    runtime_yaml = _load_yaml(ROOT / "config" / "runtime.yaml")
    runtime = RuntimeConfig(**runtime_yaml.get("runtime", {}))
    env_map = {
        "execution_device": os.getenv("EXECUTION_DEVICE"),
        "embedding_device": os.getenv("EMBEDDING_DEVICE"),
        "reranker_device": os.getenv("RERANKER_DEVICE"),
        "vector_device": os.getenv("VECTOR_DEVICE"),
    }
    runtime = runtime.model_copy(update={k: v for k, v in env_map.items() if v})
    return Settings(
        runtime=runtime,
        llm_provider=os.getenv("LLM_PROVIDER", models.get("llm", {}).get("provider", "ollama")),
        ollama_profile=os.getenv("OLLAMA_PROFILE", "balanced"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", models.get("llm", {}).get("base_url", "http://localhost:11434")),
        ollama_model=os.getenv("OLLAMA_MODEL", models.get("llm", {}).get("model", "rag-qwen-balanced")),
        ollama_connect_timeout_seconds=float(os.getenv("OLLAMA_CONNECT_TIMEOUT_S", models.get("llm", {}).get("connect_timeout_seconds", 10))),
        ollama_read_timeout_seconds=float(os.getenv("OLLAMA_READ_TIMEOUT_S", models.get("llm", {}).get("read_timeout_seconds", 600))),
        ollama_max_retries=int(os.getenv("OLLAMA_MAX_RETRIES", models.get("llm", {}).get("max_retries", 1))),
        ollama_num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", models.get("llm", {}).get("num_predict", 384))),
        embedding_model=models.get("embedding", {}).get("default", "sentence-transformers/all-MiniLM-L6-v2"),
        multilingual_embedding_model=models.get("embedding", {}).get("multilingual", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        e5_embedding_model=models.get("embedding", {}).get("e5_multilingual", "intfloat/multilingual-e5-small"),
        reranker_model=models.get("reranker", {}).get("default", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        max_context_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "1800")),
    )

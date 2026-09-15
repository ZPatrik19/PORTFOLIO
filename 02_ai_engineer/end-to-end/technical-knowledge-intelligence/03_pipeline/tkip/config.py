"""Configuration loading, validation and repository-relative path handling."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .exceptions import ConfigurationError

def _discover_project_root() -> Path:
    """Locate the runtime project root in source, editable and wheel installs.

    A wheel install lives under site-packages, so deriving the repository root
    from ``__file__`` alone is incorrect. Prefer an explicit environment
    override, then the process working directory (Docker uses ``/app``), and
    finally source-checkout ancestors containing ``config.yaml``.
    """

    explicit = os.getenv("TKI_PROJECT_ROOT", "").strip()
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if (candidate / "config.yaml").exists():
            return candidate

    search_roots = [Path.cwd().resolve(), Path(__file__).resolve().parent]
    seen: set[Path] = set()
    for start in search_roots:
        for candidate in (start, *start.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / "config.yaml").exists():
                return candidate

    # Source-tree fallback kept for helpful error messages when config.yaml is
    # missing. In an installed wheel this is intentionally only a fallback.
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _discover_project_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


class FlexibleModel(BaseModel):
    """Base model that validates known fields while preserving extensions."""

    model_config = ConfigDict(extra="allow")


class GeminiConfig(FlexibleModel):
    enabled: bool = True
    model: str = "gemini-3.8-flash"
    temperature: float = Field(default=0.45, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1800, ge=128)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=8)


class EmbeddingConfig(FlexibleModel):
    provider: Literal["auto", "gemini", "local_hashing"] = "auto"
    model: str = "gemini-embedding-2"
    output_dimensionality: int = Field(default=768, ge=64)
    batch_size: int = Field(default=16, ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=8)


class ChunkingConfig(FlexibleModel):
    strategy: Literal["fixed", "recursive", "structure_aware", "semantic"] = "structure_aware"
    chunk_size: int = Field(default=900, ge=100)
    overlap: int = Field(default=120, ge=0)
    min_chunk_chars: int = Field(default=120, ge=1)
    max_chunk_chars: int = Field(default=5000, ge=100)

    @model_validator(mode="after")
    def validate_overlap(self) -> ChunkingConfig:
        if self.overlap >= self.chunk_size:
            raise ValueError("chunking.overlap must be smaller than chunking.chunk_size")
        if self.min_chunk_chars > self.max_chunk_chars:
            raise ValueError("chunking.min_chunk_chars must not exceed max_chunk_chars")
        return self


class RetrievalConfig(FlexibleModel):
    bm25_k: int = Field(default=20, ge=1)
    dense_k: int = Field(default=20, ge=1)
    hybrid_k: int = Field(default=24, ge=1)
    final_k: int = Field(default=8, ge=1)
    fusion: str = "rrf"
    rrf_k: int = Field(default=60, ge=1)
    query_rewrite: bool = False


class ContextConfig(FlexibleModel):
    max_chars: int = Field(default=16000, ge=1000)
    max_chunks_per_document: int = Field(default=3, ge=1)
    require_document_diversity: bool = True
    deduplicate_threshold: float = Field(default=0.92, ge=0.0, le=1.0)


class PathsConfig(FlexibleModel):
    user_library: str = "01_data/user_library"
    reference_docs: str = "01_data/reference_docs"
    raw: str = "01_data/raw"
    interim: str = "01_data/interim"
    processed: str = "01_data/processed"
    indexes: str = "01_data/indexes"
    evaluation: str = "01_data/evaluation"
    feedback: str = "01_data/feedback"
    results: str = "07_results"


class AppConfig(FlexibleModel):
    project: dict[str, Any] = Field(default_factory=dict)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    reranking: dict[str, Any] = Field(default_factory=dict)
    vector_store: dict[str, Any] = Field(default_factory=dict)
    parsing: dict[str, Any] = Field(default_factory=dict)
    multimodal: dict[str, Any] = Field(default_factory=dict)
    agent: dict[str, Any] = Field(default_factory=dict)
    guardrails: dict[str, Any] = Field(default_factory=dict)
    monitoring: dict[str, Any] = Field(default_factory=dict)
    logging: dict[str, Any] = Field(default_factory=lambda: {"level": "INFO"})


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(DEFAULT_ENV_PATH, override=False)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration and validate critical application sections.

    A dictionary is returned for backward compatibility with the existing
    services. Validation is centralized here so downstream code can rely on
    required types and basic invariants.
    """

    configured = os.getenv("TKI_CONFIG_PATH", "").strip()
    config_path = Path(path) if path else Path(configured) if configured else DEFAULT_CONFIG_PATH
    config_path = config_path.expanduser().resolve()
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        # Backward-compatible path aliases for projects upgraded from <=1.0.9.
        paths = raw.setdefault("paths", {})
        if "user_library" not in paths and "private_books" in paths:
            paths["user_library"] = paths["private_books"]
        if "reference_docs" not in paths and "public_docs" in paths:
            paths["reference_docs"] = paths["public_docs"]
        validated = AppConfig.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError, ValueError) as exc:
        raise ConfigurationError(f"Invalid configuration in {config_path}: {exc}") from exc

    config = validated.model_dump(mode="python")
    config["_project_root"] = str(PROJECT_ROOT)
    return config


def resolve_path(value: str | Path) -> Path:
    """Resolve a configured path relative to the repository root."""

    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def gemini_api_key() -> str | None:
    """Return the Gemini API key from the local environment without logging it."""

    _load_dotenv()
    value = os.getenv("GEMINI_API_KEY", "").strip()
    return value or None

"""Configuration loading and validation.

YAML remains the human-editable source of truth, while Pydantic models validate
critical benchmark fields before an experiment starts. ``load_yaml`` is kept
for compatibility with report/UI code that consumes provider-specific maps.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from prompt_benchmark.paths import PATHS


class ConfigurationError(ValueError):
    """Raised when project configuration is missing or invalid."""


class BenchmarkConfig(BaseModel):
    """Validated core experiment configuration."""

    model_config = ConfigDict(extra="allow")

    model: str = ""
    temperature: float | None = None
    max_output_tokens: int = Field(default=64, ge=1, le=8192)
    random_seed: int = 42
    benchmark_samples_per_class: int = Field(default=1000, ge=1)
    development_samples_per_class: int = Field(default=500, ge=1)
    few_shot_examples_per_class: int = Field(default=4, ge=1)
    mock_source_samples_per_class: int = Field(default=1800, ge=1)
    bootstrap_iterations: int = Field(default=1000, ge=100)

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return value


def _expand_env(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.getenv(value[2:-1], "")
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load UTF-8 YAML with ``${ENV_VAR}`` expansion.

    Raises a domain-specific error with the absolute path for easier debugging.
    """
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PATHS.root / config_path
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"Configuration root must be a mapping: {config_path}")
    return _expand_env(data)


def load_benchmark_config(path: str | Path | None = None) -> BenchmarkConfig:
    """Load and validate the benchmark YAML into a typed configuration object."""
    raw = load_yaml(path or PATHS.configs / "benchmark.yaml")
    try:
        return BenchmarkConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid benchmark configuration: {exc}") from exc

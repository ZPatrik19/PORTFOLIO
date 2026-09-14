"""EN: Configuration loading, Pydantic validation, and actionable failure messages.

HU: A konfigurációbetöltést, Pydantic-validációt és az érthető hibaüzeneteket ellenőrzi.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prompt_benchmark.config import ConfigurationError, load_benchmark_config, load_yaml


def test_benchmark_config_loads_and_validates() -> None:
    """EN: Loads the benchmark YAML through the typed config model and validates required values.

    HU: Betölti a benchmark YAML-t a típusos config modellen keresztül és ellenőrzi a szükséges értékeket.
    """
    cfg = load_benchmark_config()
    assert cfg.max_output_tokens > 0
    assert cfg.benchmark_samples_per_class >= 1
    assert cfg.bootstrap_iterations >= 100


def test_missing_config_has_actionable_error(tmp_path: Path) -> None:
    """EN: Ensures a missing config file produces a clear, actionable exception.

    HU: Biztosítja, hogy hiányzó config fájl esetén érthető és javítható hibaüzenet keletkezzen.
    """
    with pytest.raises(ConfigurationError, match="does not exist"):
        load_yaml(tmp_path / "missing.yaml")


def test_invalid_yaml_has_actionable_error(tmp_path: Path) -> None:
    """EN: Ensures malformed YAML produces a clear configuration error rather than an opaque stack trace.

    HU: Biztosítja, hogy hibás YAML esetén ne átláthatatlan stack trace, hanem értelmes konfigurációs hiba jelenjen meg.
    """
    path = tmp_path / "bad.yaml"
    path.write_text("x: [unterminated", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Invalid YAML"):
        load_yaml(path)

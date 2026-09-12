from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else project_root() / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def result_path(*parts: str) -> Path:
    p = project_root() / "04_results"
    for part in parts:
        p /= part
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def data_path(*parts: str) -> Path:
    p = project_root() / "01_data"
    for part in parts:
        p /= part
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

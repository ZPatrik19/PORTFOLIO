from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from prompt_benchmark.paths import PATHS

LOGGER = logging.getLogger(__name__)

PRESET_DIR = PATHS.playground_prompts
HISTORY_PATH = PATHS.results / "playground_history.jsonl"


def _slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_").lower()
    return value or "playground_prompt"


def save_generation_preset(name: str, system_prompt: str, user_template: str, metadata: dict[str, Any] | None = None) -> Path:
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    path = PRESET_DIR / f"{_slug(name)}.json"
    payload = {
        "name": name,
        "type": "generation",
        "system_prompt": system_prompt,
        "user_template": user_template,
        "metadata": metadata or {},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def list_generation_presets() -> dict[str, Path]:
    if not PRESET_DIR.exists():
        return {}
    result: dict[str, Path] = {}
    for path in sorted(PRESET_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            result[str(payload.get("name") or path.stem)] = path
        except (json.JSONDecodeError, OSError) as exc:
            LOGGER.warning("Skipping unreadable playground preset %s: %s", path, exc)
            continue
    return result


def load_generation_preset(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_playground_history(row: dict[str, Any]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(row)
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    with HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def load_playground_history(limit: int = 100) -> pd.DataFrame:
    """Load recent Playground runs, skipping malformed JSONL rows safely."""
    if not HISTORY_PATH.exists():
        return pd.DataFrame()
    try:
        lines = HISTORY_PATH.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        LOGGER.warning("Could not read playground history %s: %s", HISTORY_PATH, exc)
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            LOGGER.warning(
                "Skipping malformed playground history row %s:%d: %s",
                HISTORY_PATH,
                line_number,
                exc,
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows[-limit:][::-1])

"""Local experiment trace persistence for prompt/RAG benchmarking."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .logging_config import get_logger

LOGGER = get_logger(__name__)


class ExperimentTracker:
    def __init__(self, root: Path) -> None:
        self.directory = root / "07_results" / "experiments"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "experiment_traces.jsonl"

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        row = dict(record)
        row.setdefault(
            "experiment_id",
            f"exp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
        )
        row.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return row

    def recent(self, limit: int = 200) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        output: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines()[-limit:], start=1
        ):
            try:
                output.append(json.loads(line))
            except json.JSONDecodeError as exc:
                LOGGER.warning("Ignoring malformed experiment trace line %s: %s", line_number, exc)
        return output

from __future__ import annotations

import json
from dataclasses import dataclass

from prompt_benchmark.constants import LABELS


@dataclass(frozen=True)
class ParsedPrediction:
    label: str | None
    valid_output: bool
    valid_json: bool | None


def parse_prediction(raw_output: str, output_mode: str) -> ParsedPrediction:
    raw = (raw_output or "").strip()
    if output_mode == "json":
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return ParsedPrediction(None, False, False)
        label = obj.get("label") if isinstance(obj, dict) else None
        valid = isinstance(label, str) and label in LABELS and set(obj.keys()) == {"label"}
        return ParsedPrediction(label if valid else None, valid, True)

    normalized = raw.lower().strip().strip("`\"' .")
    valid = normalized in LABELS
    return ParsedPrediction(normalized if valid else None, valid, None)

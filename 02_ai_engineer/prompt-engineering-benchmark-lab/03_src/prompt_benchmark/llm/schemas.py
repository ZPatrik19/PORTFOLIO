from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prompt_benchmark.constants import LABELS

SUPPORT_TICKET_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": list(LABELS),
        }
    },
    "required": ["label"],
    "additionalProperties": False,
}


@dataclass
class LLMResponse:
    """Normalized response metadata used by every benchmark provider."""

    raw_output: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_seconds: float
    model: str
    provider: str = "unknown"
    error: str | None = None
    token_source: str = "provider_reported"
    latency_source: str = "wall_clock"

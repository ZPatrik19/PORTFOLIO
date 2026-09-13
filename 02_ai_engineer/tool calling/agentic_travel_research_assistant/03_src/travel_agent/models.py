from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallRecord:
    step: int
    name: str
    arguments: dict[str, Any]
    output: dict[str, Any]
    latency_ms: float
    success: bool
    error: str | None = None


@dataclass
class AgentRun:
    query: str
    answer: str
    trace: list[ToolCallRecord] = field(default_factory=list)
    total_latency_ms: float = 0.0
    model: str = "offline"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_names(self) -> list[str]:
        return [call.name for call in self.trace]

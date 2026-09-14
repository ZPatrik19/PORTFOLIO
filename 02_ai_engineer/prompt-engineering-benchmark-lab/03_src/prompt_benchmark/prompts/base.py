from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class PromptPayload:
    """Provider-neutral prompt request produced by a prompt strategy.

    ``metadata`` is benchmark-side context (for example synthetic case type or
    difficulty). Real API providers ignore it. The mock provider uses it only
    to create a deterministic prompt-sensitivity simulation, never to change
    the prompt text that a real provider would receive.
    """

    strategy_name: str
    instructions: str | None
    input_text: str
    output_mode: str = "label"  # label | json | text
    structured_output: bool = False
    reasoning_effort: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PromptStrategy(Protocol):
    name: str

    def build(self, ticket: str) -> PromptPayload:
        ...

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompt_benchmark.paths import PATHS
from prompt_benchmark.prompts.base import PromptPayload

CUSTOM_PROMPT_DIR = PATHS.custom_prompts


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "custom_prompt"


@dataclass
class CustomPromptStrategy:
    """User-authored prompt strategy usable by the same runner as built-in prompts.

    ``user_template`` must contain ``{ticket}``; the placeholder is replaced at
    runtime. Other braces can be escaped as ``{{`` and ``}}``.
    """

    display_name: str
    system_prompt: str
    user_template: str
    output_mode: str = "label"
    structured_output: bool = False
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        if "{ticket}" not in self.user_template:
            raise ValueError("A saját user prompt template-nek tartalmaznia kell a {ticket} helyőrzőt.")
        if self.output_mode not in {"label", "json"}:
            raise ValueError("output_mode csak 'label' vagy 'json' lehet.")
        self.name = f"custom_{_slugify(self.display_name)}"

    def build(self, ticket: str) -> PromptPayload:
        try:
            rendered = self.user_template.format(ticket=ticket)
        except KeyError as exc:
            raise ValueError(
                f"Ismeretlen template-helyőrző: {exc}. Csak a {{ticket}} használható; "
                "JSON kapcsos zárójeleket {{ és }} formában escape-elj."
            ) from exc
        return PromptPayload(
            strategy_name=self.name,
            instructions=self.system_prompt.strip() or None,
            input_text=rendered,
            output_mode=self.output_mode,
            structured_output=bool(self.structured_output),
            reasoning_effort=self.reasoning_effort or None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "display_name": self.display_name,
            "system_prompt": self.system_prompt,
            "user_template": self.user_template,
            "output_mode": self.output_mode,
            "structured_output": self.structured_output,
            "reasoning_effort": self.reasoning_effort,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CustomPromptStrategy":
        return cls(
            display_name=str(data.get("display_name", "custom_prompt")),
            system_prompt=str(data.get("system_prompt", "")),
            user_template=str(data.get("user_template", "{ticket}")),
            output_mode=str(data.get("output_mode", "label")),
            structured_output=bool(data.get("structured_output", False)),
            reasoning_effort=data.get("reasoning_effort") or None,
        )


def save_custom_prompt(strategy: CustomPromptStrategy, directory: Path = CUSTOM_PROMPT_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_slugify(strategy.display_name)}.json"
    path.write_text(json.dumps(strategy.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def list_custom_prompts(directory: Path = CUSTOM_PROMPT_DIR) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob("*.json"))


def load_custom_prompt(path: str | Path) -> CustomPromptStrategy:
    p = Path(path)
    return CustomPromptStrategy.from_dict(json.loads(p.read_text(encoding="utf-8")))

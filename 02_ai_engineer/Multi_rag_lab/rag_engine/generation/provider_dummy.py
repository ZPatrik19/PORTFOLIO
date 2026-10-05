from __future__ import annotations

import re


class DummyLLMProvider:
    """Deterministic offline provider used for tests and pipeline demonstrations."""

    name = "dummy"
    model_name = "deterministic-dummy"

    def __init__(self) -> None:
        self.last_metrics: dict[str, float | int | str] = {}

    def generate(self, prompt: str) -> str:
        self.last_metrics = {"ttft_ms": 0.0, "total_ms": 0.0, "output_tokens": 0, "tokens_per_second": 0.0}
        if "hipotetikus dokumentumrészletet" in prompt:
            question = prompt.split("Kérdés:", 1)[-1].split("Hipotetikus részlet:", 1)[0].strip()
            return f"{question} részletes magyarázat tünetek okok kezelés megelőzés"
        if "második keresési lekérdezést" in prompt:
            question = prompt.split("Eredeti kérdés:", 1)[-1].split("Első körös evidence:", 1)[0].strip()
            return f"{question} további részletek"
        if "Készíts " in prompt and "keresési lekérdezést" in prompt:
            question = prompt.split("Kérdés:", 1)[-1].strip()
            return f"{question}\n{question} magyarázat\n{question} részletek"
        if "Írd át a felhasználó kérdését" in prompt:
            return prompt.split("Kérdés:", 1)[-1].strip()

        # Backward-compatible English prompt handling keeps older tests stable.
        if "Generate " in prompt and "search queries" in prompt:
            question = prompt.split("Question:", 1)[-1].strip()
            return f"{question}\n{question} explanation\n{question} details"
        if "Rewrite the user question" in prompt:
            return prompt.split("Question:", 1)[-1].strip()

        if "Bizonyítékok:" in prompt:
            evidence = prompt.split("Bizonyítékok:", 1)[-1].split("\n\nVálasz:", 1)[0]
        else:
            evidence = prompt.split("Evidence:", 1)[-1].split("\n\nAnswer:", 1)[0]
        sources = re.findall(r"\[S\d+\].*?(?=\n\[S\d+\]|\Z)", evidence, flags=re.S)
        if not sources:
            return "A rendelkezésre álló dokumentumok nem tartalmaznak elegendő információt."
        first = sources[0]
        content_marker = "Tartalom:\n" if "Tartalom:\n" in first else "Content:\n"
        content = first.split(content_marker, 1)[-1].strip()
        summary = " ".join(content.split()[:55])
        return f"{summary} [S1]"

    def health(self) -> dict[str, object]:
        return {"ok": True, "provider": self.name, "model": self.model_name}

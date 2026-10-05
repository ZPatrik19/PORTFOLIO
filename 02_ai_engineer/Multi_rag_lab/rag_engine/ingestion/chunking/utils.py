from __future__ import annotations

import re
from collections.abc import Iterable


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÖŐÚÜŰ0-9])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def estimate_tokens(text: str) -> int:
    # Lightweight, deterministic estimate used for budgeting without requiring a tokenizer.
    return max(1, round(len(text.split()) * 1.3)) if text.strip() else 0


def sliding_windows(items: Iterable[str], target_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for item in items:
        if current and length + len(item) + 1 > target_chars:
            chunks.append(" ".join(current))
            current, length = [], 0
        current.append(item)
        length += len(item) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks

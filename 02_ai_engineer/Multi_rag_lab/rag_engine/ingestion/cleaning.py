from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from rag_engine.models import Document


@dataclass(frozen=True)
class CleaningStats:
    documents_before: int
    documents_after: int
    characters_before: int
    characters_after: int
    duplicates_removed: int
    empty_sections_removed: int
    repeated_edge_lines_removed: int


_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^<>]*>")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    # Only remove strings that actually look like HTML tags.  The previous
    # ``<[^>]+>`` expression also interpreted medical/math comparisons such as
    # ``< 120`` as the start of a tag and could delete thousands of characters
    # until the next ``>`` sign (for example BMI > 30).
    text = _HTML_TAG_RE.sub(" ", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def _strip_repeated_headers_footers(documents: list[Document]) -> tuple[list[Document], int]:
    """Remove short lines repeatedly appearing at page edges of the same source document."""
    groups: dict[str, list[tuple[int, Document]]] = defaultdict(list)
    for index, document in enumerate(documents):
        if document.metadata.get("page") is not None:
            groups[str(document.metadata.get("source", ""))].append((index, document))

    repeated_by_group: dict[str, set[str]] = {}
    for source, items in groups.items():
        if len(items) < 2:
            continue
        counts: Counter[str] = Counter()
        for _, document in items:
            lines = document.text.splitlines()
            edges = {line for line in [*lines[:2], *lines[-2:]] if 0 < len(line) <= 200}
            counts.update(edges)
        threshold = max(2, int(len(items) * 0.6 + 0.999))
        repeated_by_group[source] = {line for line, count in counts.items() if count >= threshold}

    output = list(documents)
    removed = 0
    for source, items in groups.items():
        repeated = repeated_by_group.get(source, set())
        if not repeated:
            continue
        for index, document in items:
            lines = document.text.splitlines()
            kept: list[str] = []
            last_index = len(lines) - 1
            for line_index, line in enumerate(lines):
                is_edge = line_index <= 1 or line_index >= max(0, last_index - 1)
                if is_edge and line in repeated:
                    removed += 1
                    continue
                kept.append(line)
            output[index] = document.model_copy(update={"text": "\n".join(kept).strip()})
    return output, removed


def clean_documents(documents: Iterable[Document], *, min_chars: int = 20) -> tuple[list[Document], CleaningStats]:
    documents = list(documents)
    before_chars = sum(len(d.text) for d in documents)
    normalized = [document.model_copy(update={"text": normalize_text(document.text)}) for document in documents]
    normalized, edge_lines_removed = _strip_repeated_headers_footers(normalized)

    seen: set[str] = set()
    cleaned: list[Document] = []
    duplicate_count = 0
    empty_count = 0
    for document in normalized:
        text = document.text.strip()
        if len(text) < min_chars:
            empty_count += 1
            continue
        fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if fingerprint in seen:
            duplicate_count += 1
            continue
        seen.add(fingerprint)
        cleaned.append(document.model_copy(update={"text": text}))

    return cleaned, CleaningStats(
        documents_before=len(documents),
        documents_after=len(cleaned),
        characters_before=before_chars,
        characters_after=sum(len(d.text) for d in cleaned),
        duplicates_removed=duplicate_count,
        empty_sections_removed=empty_count,
        repeated_edge_lines_removed=edge_lines_removed,
    )

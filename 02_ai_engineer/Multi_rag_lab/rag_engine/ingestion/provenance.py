from __future__ import annotations

from typing import Any


def chunk_metadata(
    base: dict[str, Any],
    *,
    document_id: str,
    chunk_id: str,
    strategy: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    section: str | None = None,
) -> dict[str, Any]:
    return {
        **base,
        "document_id": document_id,
        "chunk_id": chunk_id,
        "section": section or base.get("section", ""),
        "chunking_strategy": strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }

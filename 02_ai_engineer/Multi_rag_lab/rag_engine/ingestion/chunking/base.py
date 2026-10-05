from __future__ import annotations

from typing import Protocol

from rag_engine.models import Chunk, Document


class ChunkingStrategy(Protocol):
    name: str

    def chunk(self, documents: list[Document]) -> list[Chunk]: ...

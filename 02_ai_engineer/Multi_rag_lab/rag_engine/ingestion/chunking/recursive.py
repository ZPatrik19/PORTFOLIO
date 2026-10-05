from __future__ import annotations

import re

from rag_engine.ingestion.provenance import chunk_metadata
from rag_engine.models import Chunk, Document


class RecursiveChunker:
    name = "recursive"

    def __init__(self, chunk_size: int = 500, overlap: int = 100, separators: list[str] | None = None) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def _split(self, text: str, level: int = 0) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []
        separator = self.separators[min(level, len(self.separators) - 1)]
        if not separator:
            step = max(1, self.chunk_size - self.overlap)
            return [text[i:i + self.chunk_size].strip() for i in range(0, len(text), step) if text[i:i + self.chunk_size].strip()]
        pieces = text.split(separator)
        if len(pieces) == 1:
            return self._split(text, level + 1)
        output: list[str] = []
        current = ""
        for piece in pieces:
            candidate = (current + separator + piece).strip() if current else piece.strip()
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    output.extend(self._split(current, level + 1))
                current = piece.strip()
        if current:
            output.extend(self._split(current, level + 1))
        return output

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        output: list[Chunk] = []
        for doc in documents:
            pieces = self._split(doc.text)
            for index, text in enumerate(pieces):
                chunk_id = f"{doc.document_id}:recursive:{index}"
                output.append(Chunk(chunk_id=chunk_id, document_id=doc.document_id, text=text,
                    metadata=chunk_metadata(doc.metadata, document_id=doc.document_id, chunk_id=chunk_id,
                                            strategy=self.name, chunk_size=self.chunk_size, chunk_overlap=self.overlap)))
        return output

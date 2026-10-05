from __future__ import annotations

from rag_engine.ingestion.provenance import chunk_metadata
from rag_engine.models import Chunk, Document


class FixedChunker:
    name = "fixed"

    def __init__(self, chunk_size: int = 500, overlap: int = 100) -> None:
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        output: list[Chunk] = []
        step = self.chunk_size - self.overlap
        for doc in documents:
            for index, start in enumerate(range(0, len(doc.text), step)):
                text = doc.text[start : start + self.chunk_size].strip()
                if not text:
                    continue
                chunk_id = f"{doc.document_id}:fixed:{index}"
                output.append(Chunk(
                    chunk_id=chunk_id,
                    document_id=doc.document_id,
                    text=text,
                    metadata=chunk_metadata(doc.metadata, document_id=doc.document_id, chunk_id=chunk_id,
                                            strategy=self.name, chunk_size=self.chunk_size, chunk_overlap=self.overlap),
                ))
                if start + self.chunk_size >= len(doc.text):
                    break
        return output


class FixedTokenChunker:
    """Whitespace-token baseline. Tokenizer-specific chunking can be added behind the same interface."""

    name = "fixed-token"

    def __init__(self, chunk_size: int = 200, overlap: int = 40) -> None:
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        output: list[Chunk] = []
        step = self.chunk_size - self.overlap
        for doc in documents:
            tokens = doc.text.split()
            for index, start in enumerate(range(0, len(tokens), step)):
                text = " ".join(tokens[start : start + self.chunk_size]).strip()
                if not text:
                    continue
                chunk_id = f"{doc.document_id}:fixed-token:{index}"
                output.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=doc.document_id,
                        text=text,
                        metadata=chunk_metadata(
                            doc.metadata,
                            document_id=doc.document_id,
                            chunk_id=chunk_id,
                            strategy=self.name,
                            chunk_size=self.chunk_size,
                            chunk_overlap=self.overlap,
                        ),
                    )
                )
                if start + self.chunk_size >= len(tokens):
                    break
        return output

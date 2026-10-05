from __future__ import annotations

from rag_engine.ingestion.chunking.utils import sliding_windows, split_sentences
from rag_engine.ingestion.provenance import chunk_metadata
from rag_engine.models import Chunk, Document


class SentenceChunker:
    name = "sentence"

    def __init__(self, chunk_size: int = 500) -> None:
        self.chunk_size = chunk_size

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        output: list[Chunk] = []
        for doc in documents:
            parts = sliding_windows(split_sentences(doc.text), self.chunk_size)
            for index, text in enumerate(parts):
                chunk_id = f"{doc.document_id}:sentence:{index}"
                output.append(Chunk(chunk_id=chunk_id, document_id=doc.document_id, text=text,
                    metadata=chunk_metadata(doc.metadata, document_id=doc.document_id, chunk_id=chunk_id,
                                            strategy=self.name, chunk_size=self.chunk_size)))
        return output

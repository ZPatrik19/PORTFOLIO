from __future__ import annotations

import re

from rag_engine.ingestion.chunking.utils import sliding_windows
from rag_engine.ingestion.provenance import chunk_metadata
from rag_engine.models import Chunk, Document


class ParagraphChunker:
    name = "paragraph"

    def __init__(self, chunk_size: int = 700) -> None:
        self.chunk_size = chunk_size

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        output: list[Chunk] = []
        for doc in documents:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n(?=[A-ZÁÉÍÓÖŐÚÜŰ])", doc.text) if p.strip()]
            for index, text in enumerate(sliding_windows(paragraphs, self.chunk_size)):
                chunk_id = f"{doc.document_id}:paragraph:{index}"
                output.append(Chunk(chunk_id=chunk_id, document_id=doc.document_id, text=text,
                    metadata=chunk_metadata(doc.metadata, document_id=doc.document_id, chunk_id=chunk_id,
                                            strategy=self.name, chunk_size=self.chunk_size)))
        return output

from __future__ import annotations

import re

from rag_engine.ingestion.chunking.recursive import RecursiveChunker
from rag_engine.ingestion.provenance import chunk_metadata
from rag_engine.models import Chunk, Document


class StructuralChunker:
    name = "structure-aware"

    def __init__(self, chunk_size: int = 700) -> None:
        self.chunk_size = chunk_size
        self.fallback = RecursiveChunker(chunk_size=chunk_size, overlap=0)

    def _fallback_chunks(self, doc: Document) -> list[Chunk]:
        output: list[Chunk] = []
        for index, piece in enumerate(self.fallback._split(doc.text)):
            chunk_id = f"{doc.document_id}:structural:fallback:{index}"
            output.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=doc.document_id,
                    text=piece,
                    metadata=chunk_metadata(
                        doc.metadata,
                        document_id=doc.document_id,
                        chunk_id=chunk_id,
                        strategy=self.name,
                        chunk_size=self.chunk_size,
                    ),
                )
            )
        return output

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        output: list[Chunk] = []
        heading_re = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
        for doc in documents:
            matches = list(heading_re.finditer(doc.text))
            if not matches:
                output.extend(self._fallback_chunks(doc))
                continue

            for index, match in enumerate(matches):
                start = match.end()
                end = matches[index + 1].start() if index + 1 < len(matches) else len(doc.text)
                heading = match.group(2).strip()
                section_text = doc.text[start:end].strip()
                if not section_text:
                    continue
                section_doc = doc.model_copy(
                    update={"text": section_text, "metadata": {**doc.metadata, "section": heading}}
                )
                for local_index, piece in enumerate(self.fallback._split(section_text)):
                    chunk_id = f"{doc.document_id}:structural:{index}:{local_index}"
                    output.append(
                        Chunk(
                            chunk_id=chunk_id,
                            document_id=doc.document_id,
                            text=piece,
                            metadata=chunk_metadata(
                                section_doc.metadata,
                                document_id=doc.document_id,
                                chunk_id=chunk_id,
                                strategy=self.name,
                                chunk_size=self.chunk_size,
                                section=heading,
                            ),
                        )
                    )
        return output

from __future__ import annotations

import numpy as np

from rag_engine.ingestion.chunking.utils import split_sentences
from rag_engine.ingestion.provenance import chunk_metadata
from rag_engine.models import Chunk, Document


class SemanticChunker:
    name = "semantic"

    def __init__(self, embedder, threshold: float = 0.72, max_chars: int = 1200) -> None:
        self.embedder = embedder
        self.threshold = threshold
        self.max_chars = max_chars

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        output: list[Chunk] = []
        for doc in documents:
            sentences = split_sentences(doc.text)
            if not sentences:
                continue
            vectors = self.embedder.embed_documents(sentences)
            groups: list[list[str]] = [[sentences[0]]]
            for index in range(1, len(sentences)):
                a, b = vectors[index - 1], vectors[index]
                denom = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
                similarity = float(np.dot(a, b) / denom)
                current_len = sum(len(x) for x in groups[-1])
                if similarity < self.threshold or current_len + len(sentences[index]) > self.max_chars:
                    groups.append([sentences[index]])
                else:
                    groups[-1].append(sentences[index])
            for index, group in enumerate(groups):
                text = " ".join(group)
                chunk_id = f"{doc.document_id}:semantic:{index}"
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
                            chunk_size=self.max_chars,
                        ),
                    )
                )
        return output

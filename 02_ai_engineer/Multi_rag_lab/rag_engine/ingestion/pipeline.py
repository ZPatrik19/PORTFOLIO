from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from rag_engine.ingestion.cleaning import CleaningStats, clean_documents
from rag_engine.ingestion.parser import parse_file
from rag_engine.models import ChunkingConfig
from rag_engine.models import Chunk, Document
from rag_engine.ingestion.chunking.factory import create_chunker


@dataclass
class IngestionResult:
    documents: list[Document]
    chunks: list[Chunk]
    cleaning: CleaningStats

    def summary(self) -> dict[str, object]:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "cleaning": asdict(self.cleaning),
        }


def ingest_paths(paths: list[Path], config: ChunkingConfig, *, embedder=None) -> IngestionResult:
    raw_documents: list[Document] = []
    for path in paths:
        raw_documents.extend(parse_file(path))
    documents, stats = clean_documents(raw_documents)
    chunker = create_chunker(config, embedder=embedder)
    chunks = chunker.chunk(documents)
    return IngestionResult(documents=documents, chunks=chunks, cleaning=stats)

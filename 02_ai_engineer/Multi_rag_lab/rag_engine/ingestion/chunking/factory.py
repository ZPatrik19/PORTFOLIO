from __future__ import annotations

from rag_engine.ingestion.chunking.fixed import FixedChunker, FixedTokenChunker
from rag_engine.ingestion.chunking.paragraph import ParagraphChunker
from rag_engine.ingestion.chunking.parent_child import ParentChildChunker
from rag_engine.ingestion.chunking.recursive import RecursiveChunker
from rag_engine.ingestion.chunking.semantic import SemanticChunker
from rag_engine.ingestion.chunking.sentence import SentenceChunker
from rag_engine.ingestion.chunking.structural import StructuralChunker
from rag_engine.models import ChunkingConfig


def create_chunker(config: ChunkingConfig, *, embedder=None):
    name = config.strategy.lower()
    if name == "fixed":
        return FixedChunker(config.chunk_size, config.chunk_overlap)
    if name in {"fixed-token", "fixed_token"}:
        return FixedTokenChunker(config.chunk_size, config.chunk_overlap)
    if name == "recursive":
        return RecursiveChunker(config.chunk_size, config.chunk_overlap)
    if name == "sentence":
        return SentenceChunker(config.chunk_size)
    if name == "paragraph":
        return ParagraphChunker(config.chunk_size)
    if name in {"structure-aware", "structural"}:
        return StructuralChunker(config.chunk_size)
    if name in {"parent-child", "parent_child"}:
        return ParentChildChunker(max(config.chunk_size * 3, 600), config.chunk_size)
    if name == "semantic":
        if embedder is None:
            raise ValueError("Semantic chunking requires an embedding provider")
        return SemanticChunker(embedder, config.semantic_threshold, max_chars=config.chunk_size * 2)
    raise ValueError(f"Unknown chunking strategy: {config.strategy}")

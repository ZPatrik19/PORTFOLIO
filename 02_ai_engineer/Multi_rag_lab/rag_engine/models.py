from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ExecutionDevice(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class RuntimeConfig(BaseModel):
    execution_device: ExecutionDevice = ExecutionDevice.AUTO
    embedding_device: ExecutionDevice = ExecutionDevice.AUTO
    reranker_device: ExecutionDevice = ExecutionDevice.AUTO
    vector_device: ExecutionDevice = ExecutionDevice.CPU
    allow_cpu_fallback: bool = True
    max_oom_retries: int = Field(default=2, ge=0, le=5)


class ChunkingConfig(BaseModel):
    strategy: str = "recursive"
    chunk_size: int = Field(default=500, ge=50)
    chunk_overlap: int = Field(default=100, ge=0)
    semantic_threshold: float = Field(default=0.72, ge=-1.0, le=1.0)


class RetrievalConfig(BaseModel):
    top_k: int = Field(default=5, ge=1)
    candidate_count: int = Field(default=20, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    dense_weight: float = Field(default=0.5, ge=0, le=1)


class GenerationConfig(BaseModel):
    max_context_tokens: int = Field(default=1800, ge=128)
    temperature: float = Field(default=0.1, ge=0, le=2)


class Document(BaseModel):
    document_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    parent_id: str | None = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    source: str = ""
    score: float
    rank: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGResult(BaseModel):
    query: str
    transformed_queries: list[str] = Field(default_factory=list)
    answer: str
    citations: list[str] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    reranking_latency_ms: float | None = None
    generation_latency_ms: float = 0.0
    generation_ttft_ms: float | None = None
    output_tokens: int | None = None
    tokens_per_second: float | None = None
    total_latency_ms: float = 0.0
    context_tokens: int = 0
    execution_device: str = "cpu"
    trace: dict[str, float | str] = Field(default_factory=dict)
    context_text: str = ""
    grounded_prompt: str = ""
    grounded_prompt_parts: dict[str, str] = Field(default_factory=dict)

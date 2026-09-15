from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    title: str
    author: str | None = None
    edition: str | None = None
    document_type: str
    source: str
    source_type: Literal["private", "public", "demo"]
    language: str = "unknown"
    page_count: int = 0
    ingestion_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    checksum: str
    document_version: str = "1"
    path: str


class ParsedBlock(BaseModel):
    document_id: str
    page: int
    block_type: Literal[
        "heading",
        "paragraph",
        "list",
        "code",
        "table",
        "caption",
        "figure",
        "equation",
        "header_footer",
        "unknown",
    ]
    text: str
    chapter: str | None = None
    section: str | None = None
    programming_language: str | None = None
    asset_path: str | None = None


class Chunk(BaseModel):
    document_id: str
    chunk_id: str
    title: str
    author: str | None = None
    chapter: str | None = None
    section: str | None = None
    page_start: int = 0
    page_end: int = 0
    chunk_type: str = "paragraph"
    language: str = "unknown"
    programming_language: str | None = None
    framework: str | None = None
    keywords: list[str] = Field(default_factory=list)
    source: str
    source_type: str = "public"
    asset_path: str | None = None
    text: str


class SearchHit(BaseModel):
    chunk: Chunk
    rank: int
    bm25_score: float | None = None
    dense_score: float | None = None
    hybrid_score: float | None = None
    reranker_score: float | None = None


class SourceCitation(BaseModel):
    document_id: str
    document_title: str
    page: int | None = None
    chapter: str | None = None
    section: str | None = None
    chunk_id: str
    quote_or_evidence: str


class DiagramNode(BaseModel):
    id: str
    label: str


class DiagramEdge(BaseModel):
    source: str
    target: str
    label: str | None = None


class AnswerDiagram(BaseModel):
    title: str
    nodes: list[DiagramNode] = Field(default_factory=list)
    edges: list[DiagramEdge] = Field(default_factory=list)


class SourceVisual(BaseModel):
    document_id: str
    document_title: str
    page: int | None = None
    kind: str = "page_preview"
    asset_path: str
    caption: str | None = None


class KnowledgeAnswer(BaseModel):
    answer: str
    confidence: float = Field(ge=0, le=1)
    sources: list[SourceCitation] = Field(default_factory=list)
    used_documents: list[str] = Field(default_factory=list)
    used_tools: list[str] = Field(default_factory=list)
    answer_type: str = "grounded"
    insufficient_evidence: bool = False
    follow_up_questions: list[str] = Field(default_factory=list)
    diagram: AnswerDiagram | None = None
    visuals: list[SourceVisual] = Field(default_factory=list)
    request_id: str | None = None
    trace_id: str | None = None
    latency_ms: float | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=5000)
    language: Literal["auto", "en", "hu"] = "auto"
    mode: Literal["ask", "deep_search", "compare", "code", "learning"] = "ask"
    document_ids: list[str] = Field(default_factory=list)
    topic: str | None = None
    source_type: str | None = None
    chunk_type: str | None = None
    debug: bool = False

    # Interactive prompt controls. These are deliberately request-scoped and are
    # never persisted into the corpus/index configuration.
    prompt_style: Literal["grounded", "teacher", "comparison", "code_first", "custom"] = "grounded"
    custom_system_prompt: str | None = Field(default=None, max_length=12000)
    custom_instructions: str | None = Field(default=None, max_length=8000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    prompt_language_check: bool = True
    quality_review: bool = True

    # Optional advanced prompt-engineering rewrite. The optimizer transforms the
    # user request before query understanding/retrieval while preserving the
    # original question for diagnostics and QA.
    prompt_optimization: Literal[
        "none",
        "rag_grounded",
        "concise_expert",
        "technical_deep_dive",
        "structured_tutor",
        "socratic_tutor",
        "comparison_matrix",
        "code_first",
        "debug_root_cause",
        "system_architecture",
        "production_readiness",
        "mathematical_derivation",
        "research_synthesis",
        "decision_tradeoff",
        "costar",
        "crispe",
        "evidence_verification",
    ] = "none"
    prompt_optimization_use_gemini: bool = True

    # Select a persistent embedding/index variant. ``primary`` is the standard
    # production index; the others are independent chunking experiments.
    index_variant: Literal["primary", "fixed", "recursive", "structure_aware", "semantic"] = (
        "primary"
    )

    # Query-time context re-chunking. The persistent retrieval index is not
    # rebuilt; only the retrieved candidates are reshaped before context build.
    runtime_chunking: bool = False
    runtime_chunk_strategy: Literal["fixed", "recursive", "structure_aware", "semantic"] = (
        "structure_aware"
    )
    runtime_chunk_size: int = Field(default=900, ge=250, le=3000)
    runtime_chunk_overlap: int = Field(default=120, ge=0, le=1000)

    # Request-level cost/quality budget controls used by presets and A/B tests.
    answer_preset: Literal["economy", "recommended", "deep", "max_quality", "custom"] = (
        "recommended"
    )
    context_max_chars: int | None = Field(default=None, ge=3000, le=60000)
    retrieval_final_k: int | None = Field(default=None, ge=3, le=20)
    max_output_tokens: int | None = Field(default=None, ge=300, le=8000)
    max_tool_calls: int | None = Field(default=None, ge=0, le=8)
    include_source_visual: bool = True


class FeedbackRecord(BaseModel):
    request_id: str
    helpful: bool
    feedback_text: str | None = None

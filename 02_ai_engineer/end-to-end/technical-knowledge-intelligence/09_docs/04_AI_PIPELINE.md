# AI / RAG Pipeline

## Query path

1. Input guardrails.
2. Optional language cleanup.
3. Optional prompt-profile transformation.
4. Intent/topic/context analysis.
5. Persistent index selection.
6. BM25 + dense retrieval.
7. Reciprocal Rank Fusion.
8. Reranking.
9. Optional query-time re-chunking.
10. Context deduplication, diversity and budget control.
11. Allowlisted tool routing/execution.
12. Gemini structured synthesis.
13. Citation/output validation.
14. Optional trusted source visual extraction.
15. Quality review and telemetry.

## Retrieval and reranking

BM25 is strong for exact technical identifiers; dense retrieval handles semantic paraphrases. RRF combines rankings without assuming BM25 and cosine scores share a calibrated scale. The candidate stage deliberately retrieves more chunks than the final context needs; reranking then improves ordering and context precision.

## Context engineering

`ContextBuilder` enforces a bounded context, removes near-duplicates and limits chunks per document. This reduces repetitive evidence and keeps comparison questions from being dominated by one source.

## Prompt engineering

Prompt optimization is a separate stage and is not allowed to answer the question itself. Profiles make the requested response shape explicit—for example grounded RAG, concise expert, technical deep dive, tutor, comparison, CO-STAR or CRISPE.

## Tool calling

Gemini can propose only declarations registered by `ToolRegistry`. The backend validates tool name and arguments before execution. Model intent is therefore not treated as authorization.

## Guardrails and no-answer behavior

Retrieved documents are treated as untrusted data, full-document extraction requests are constrained, citation IDs are validated, and insufficient evidence produces an explicit abstention path. These controls reduce risk but are not a claim of perfect prompt-injection resistance.

## Gemini integration

`GeminiService` owns provider calls, retry behavior, prompt optimization and structured generation. Keys come from the session or local environment and are never committed. Provider quota/auth errors are surfaced explicitly rather than silently converted into a fabricated answer.

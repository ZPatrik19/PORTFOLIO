# Project Overview

## Problem

Technical knowledge is fragmented across user-owned programming/AI books, notes and reference documentation. A plain chatbot cannot reliably show where an answer came from, compare retrieval strategies, expose ranking behavior or diagnose failures.

## Objective

Build a local-first technical knowledge intelligence platform that ingests user-library and reference documents and provides multilingual, source-grounded answers with inspectable retrieval, tool calling, citations, evaluation and monitoring.

## Engineering objective

The portfolio objective is to demonstrate production-oriented AI engineering rather than only LLM prompting: document processing, caching, embeddings, hybrid retrieval, reranking, context engineering, structured generation, tool execution, API/UI serving, observability, evaluation and regression controls.

## Approach

1. Discover/version documents with stable IDs and checksums.
2. Parse logical structures from PDF/DOCX/Markdown/HTML and optional EPUB.
3. Compare multiple chunking strategies.
4. Build lexical and embedding indexes with cache reuse.
5. Retrieve with BM25 + dense search and RRF fusion.
6. Rerank, deduplicate and budget grounded context.
7. Optionally call allowlisted backend tools.
8. Generate structured Gemini answers and validate citations.
9. Store telemetry, feedback and experiment traces.
10. Evaluate retrieval, citations, abstention, tools, robustness and performance.

## Technologies and why

- **Python/Pydantic** — typed domain contracts and validation.
- **PyMuPDF/python-docx/BeautifulSoup** — practical multi-format parsing.
- **NumPy local vector store** — deterministic, portable default without infrastructure dependency.
- **BM25 + dense + RRF** — lexical exact-match and semantic retrieval complement one another.
- **Gemini** — structured grounded synthesis and optional function calling.
- **FastAPI** — external typed serving boundary.
- **Streamlit** — inspectable portfolio/research UI.
- **SQLite + JSONL** — lightweight local telemetry without unnecessary infrastructure.
- **pytest** — deterministic software/evaluation/regression layers.

## Limitations

- Live Gemini quality depends on provider quota/model availability.
- The generated evaluation set is not a human-curated benchmark.
- Local hashing embeddings are deterministic but weaker than production multilingual embeddings.
- PDF logical structure extraction remains heuristic for complex publisher layouts.
- Kubernetes manifests are reference deployment artifacts; actual cluster sizing depends on corpus/index size.

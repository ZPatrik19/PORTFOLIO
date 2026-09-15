# LangChain and FastAPI Integration

## Design principle

The native `HybridRetriever`, reranker, context builder, citation validator and evaluation code remain the source of truth. This keeps retrieval math inspectable and benchmarkable. LangChain is an optional adapter rather than the hidden implementation of the system.

## LangChain

`03_pipeline/tkip/langchain_adapter.py` exposes the native retriever as a LangChain Core `BaseRetriever` and provides an optional Runnable that returns:

```text
query
  ↓
TKIHybridRetriever
  ↓
LangChain Document[]
  ↓
Runnable context adapter
  ↓
{documents, context, citations}
```

This makes the portfolio engine compatible with a wider LangChain ecosystem without losing the custom BM25 + dense + RRF + reranking implementation.

Example:

```python
from tkip.langchain_adapter import create_langchain_retriever

retriever = create_langchain_retriever(platform.retriever, k=8)
documents = retriever.invoke("How does RAG work?")
```

## FastAPI

FastAPI is the service boundary around the same `KnowledgePlatform` used by the UI.

The API now uses:

- lifespan loading for one shared knowledge index per process;
- dependency injection for the platform instance;
- request/trace middleware;
- Pydantic request/response contracts;
- request-scoped `X-Gemini-API-Key` support without persistence;
- endpoints for multi-index discovery/building, prompt optimization and workflow inspection.

Important endpoints:

```text
GET  /workflow
GET  /indexes
POST /indexes/build
GET  /prompt/profiles
POST /prompt/optimize
GET  /langchain/status
POST /search
POST /ask
GET  /metrics
```

The Streamlit UI can remain an interactive research frontend, while FastAPI demonstrates how the same AI system can be consumed by another application or frontend.

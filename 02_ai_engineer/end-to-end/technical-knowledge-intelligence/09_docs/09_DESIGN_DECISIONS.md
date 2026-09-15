# Design Decisions

## Installable package instead of `sys.path` hacks
**Decision:** install `tkip` as an editable/package dependency.  
**Reason:** the same import model works in pytest, notebooks, CLI, Docker, FastAPI and Streamlit.

## NumPy as default vector backend
**Decision:** keep a portable local NumPy vector index as the default; Qdrant remains optional.  
**Reason:** the portfolio must run without infrastructure. Qdrant is useful when operational filtering/scaling justifies it.

## Native retriever is the source of truth
**Decision:** LangChain is an adapter, not the retrieval implementation.  
**Reason:** ranking logic, scores and benchmarks remain transparent and directly testable.

## Separate index lifecycle from request orchestration
**Decision:** `IndexService` owns ingestion/build/load; `KnowledgePlatform` owns request execution.  
**Reason:** reduces the God Object problem and allows independent tests.

## Configuration validation at startup
**Decision:** validate critical YAML sections with Pydantic and return a backward-compatible dict.  
**Reason:** invalid chunk/retrieval settings fail early with actionable errors without forcing a risky full rewrite.

## Do not fake semantic quality metrics
**Decision:** deterministic evaluation leaves faithfulness/answer correctness/hallucination as unknown unless a valid judge/reference exists.  
**Reason:** a convincing portfolio should distinguish measured evidence from estimates.

## UI and API are parallel adapters
**Decision:** both adapters call the same core package; Streamlit does not need an HTTP hop through FastAPI.  
**Reason:** keeps local latency/complexity low while still exposing a production-style API for external consumers.

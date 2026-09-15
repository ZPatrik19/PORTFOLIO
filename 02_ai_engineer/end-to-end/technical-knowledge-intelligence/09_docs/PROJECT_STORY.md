# Project Story — Interview Narrative

## Problem

I wanted to turn a private technical-book collection into a searchable AI knowledge system where answers are grounded in real source material and the retrieval process is measurable. A simple LLM chat interface was not enough because it hides whether the right evidence was retrieved, whether citations are valid, why an answer failed and how cost/latency change across configurations.

## Challenge

The difficult part was not calling an LLM API. The system had to handle heterogeneous technical documents, preserve structure, produce reusable chunks, combine lexical and semantic retrieval, control context size, execute tools safely, generate structured answers, validate citations, survive external API failures and provide reproducible evaluation.

A second challenge was productionization: the project evolved through multiple experiments and accumulated large UI/orchestration modules, local path workarounds and test/deployment drift. The final engineering task was therefore to refactor without losing working functionality.

## Approach

I split the problem into two independent lifecycles.

### Index lifecycle

```text
files → versioning → parsing → quality → chunking → embeddings → indexes
```

### Query lifecycle

```text
question → guardrails → query/prompt analysis → retrieval → RRF → reranking
→ context engineering → tools → Gemini → citation validation → telemetry
```

This separation made it possible to benchmark retrieval without requiring generation and to test most of the system without consuming LLM quota.

## Architecture

The reusable domain logic lives in the installable `tkip` package. `IndexService` owns ingestion/index state, while `KnowledgePlatform` coordinates a user request. Streamlit and FastAPI are parallel adapters over the same core service.

The native retrieval engine remains the source of truth. LangChain is optional because hiding retrieval behind a framework would make ranking behavior and metrics harder to explain in an interview or code review.

## Engineering Decisions

### Hybrid retrieval instead of one retrieval method

BM25 is strong on literal identifiers, commands and API names. Dense retrieval handles paraphrases. Reciprocal Rank Fusion combines their rankings without pretending that incompatible raw score scales are calibrated.

### NumPy as the local default

A portable local vector matrix keeps the project runnable without infrastructure. Qdrant is available as an optional service backend when operational filtering/scaling justifies it.

### Explicit abstention and citation validation

The platform should sometimes say that the corpus does not contain sufficient evidence. Citations are validated against the selected context instead of trusting model-generated source identifiers.

### Local/offline tests separated from live LLM tests

Provider quota should not make CI unreliable. Retrieval, metrics, regression, robustness and most orchestration behavior are deterministic and can run offline. Live Gemini tests are explicit opt-in tests.

### Do not fabricate AI metrics

Faithfulness, hallucination and answer correctness are only reported when a valid judge/reference exists. The deterministic evaluation suite does not invent these scores for portfolio optics.

## Evaluation

I evaluate the system at multiple levels:

- software correctness with unit/integration/smoke tests;
- retrieval quality with Recall@K, Precision@K, MRR, nDCG and Hit Rate;
- system performance with latency percentiles;
- citations and abstention deterministically;
- tool execution separately;
- regression and robustness with dedicated datasets/tests;
- live generation quality only when provider access and a valid judge/reference are available.

The release suite currently has 103 deterministic/offline tests passing with 61% branch-aware core coverage. The demo benchmark also produces reproducible retrieval results and failure reports.

## Results

The final repository is no longer a notebook/chatbot demo. It contains:

- installable Python package;
- versioned ingestion/index pipeline;
- hybrid retrieval and reranking;
- structured Gemini generation;
- tool calling and citation validation;
- benchmark/evaluation framework;
- telemetry and feedback loop;
- FastAPI and Streamlit adapters;
- Windows/Linux runners;
- Docker/Kubernetes deployment assets;
- CI configuration;
- layered test system and technical documentation.

## Lessons Learned

The most important lesson was that AI quality problems are often pipeline problems rather than model problems. A bad answer may come from retrieval, reranking, context construction, prompt design, tool routing, provider failure or generation. Measuring those stages separately makes debugging much more effective.

The second lesson was that productionization is mostly about explicit contracts and failure behavior: configuration validation, package boundaries, reproducible paths, quota-aware API handling, deterministic tests, health checks, deployment constraints and truthful evaluation claims.

## Productionization

To move from experiment to production-oriented system I added/refined:

- package installation instead of import hacks;
- configuration validation and `.env` secret separation;
- structured logging and telemetry;
- domain-specific exceptions;
- health/readiness endpoints;
- deterministic test/evaluation layers;
- cross-platform runners;
- non-root Docker image;
- Kubernetes probes/resources/secret example;
- CI quality/test workflow;
- technical audit, architecture, testing, deployment and troubleshooting documentation.

The resulting portfolio story is therefore not “I connected Gemini to PDFs.” It is: **I designed, implemented, benchmarked, validated and productionized an inspectable technical knowledge intelligence platform.**

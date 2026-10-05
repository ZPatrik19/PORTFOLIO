# RAG pipeline-ok

## Standard online pipeline

```text
User query
  ↓
Query processing
  ↓
Retriever
  ├─ dense
  ├─ BM25
  └─ hybrid
       ↓
Fusion
  ↓
Candidate pool
  ↓
Reranker
  ↓
Context Builder
  ↓
Grounded Generator
  ↓
Validator / Repair / Fallback
  ↓
Citations + Answer
```

## Stratégiák

- **baseline** – dense retrieval + generation;
- **hybrid** – dense + BM25 fusion;
- **lexical** – BM25-only baseline;
- **reranked** – hybrid candidate pool + reranker;
- **dense-reranked** – dense candidate pool + reranker;
- **query-rewrite** – LLM query transformation a retrieval előtt;
- **multi-query** – több query eredményeinek RRF fúziója;
- **HyDE** – hipotetikus passage embedding-alapú kereséshez;
- **multi-hop** – első evidence alapján második retrieval hop;
- **parent-document** – child retrieval után parent context;
- **compression** – query-releváns sentence compression;
- **corrective** – relevance check + bounded rewrite loop.

A strategy implementációk közös `BasePipeline` lifecycle-t használnak, így a latency tracing és generation behavior összevethető.

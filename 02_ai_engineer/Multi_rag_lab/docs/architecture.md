# Architecture

## Tervezési cél

A backend modulhatárai a RAG adatfolyam nagy mérnöki lépéseit követik. Az egyes mappák nem frameworkök vagy véletlenszerű technikai részletek szerint, hanem felelősségi kör szerint szerveződnek.

```text
ingestion → indexing → retrieval → generation → evaluation
                         ↑                  ↑
                       platform ────────────┘
```

## 1. Ingestion

`rag_engine/ingestion/`

Feladata a nyers dokumentumokból normalizált `Document` és `Chunk` objektumok előállítása.

Fő lépések:

```text
download/catalog
      ↓
parser
      ↓
cleaning
      ↓
provenance
      ↓
chunking
      ↓
ingestion pipeline
```

A `chunking/` külön alcsomag maradt, mert nyolc önálló algoritmust tartalmaz, és ez a projekt egyik fő demonstrációs területe.

## 2. Indexing

`rag_engine/indexing/`

Egy helyen kezeli a vektorizálást és a vektorindexet:

- sentence-transformer embedding;
- multilingual/E5 modellek;
- hashing fallback;
- FAISS CPU;
- FAISS GPU;
- NumPy fallback.

Az embedding és vector-store korábban külön top-level backend mappák voltak; most ugyanabban az indexing domainben vannak.

## 3. Retrieval

`rag_engine/retrieval/`

Ez az online RAG legnagyobb egysége:

- query focus;
- query rewrite / Multi-Query / HyDE / follow-up query;
- dense retrieval;
- BM25;
- hybrid fusion;
- RRF / weighted fusion;
- lexical/Cross-Encoder reranking;
- context building;
- 12 RAG strategy orchestration.

A korábbi `query/`, `relevance/`, `retrieval/`, `reranking/`, `context/` és `rag/` mappák egy domain alá kerültek. A sok 10–30 soros RAG strategy fájl `strategies.py`-ba, a query transformationök pedig `query_transform.py`-ba lettek összevonva.

## 4. Generation

`rag_engine/generation/`

Felelőssége:

- LLM provider adapterek;
- prompt construction;
- grounded answer synthesis;
- output validation;
- retry/repair;
- deterministic evidence fallback.

A generation nem végez retrievalt és nem épít indexet.

## 5. Evaluation

`rag_engine/evaluation/`

Ide kerül minden olyan logika, amely mér, benchmarkol vagy összehasonlít:

- retrieval metrics;
- generation metrics;
- medical evaluation dataset;
- pipeline matrix;
- runtime estimation;
- CPU/CUDA device benchmark;
- projection/visualization adat-előkészítés.

## 6. Platform

`rag_engine/platform/`

Keresztmetszeti technikai szolgáltatások:

- config;
- runtime/device resolution;
- hardware detection;
- memory management;
- Ollama/FAISS/CUDA diagnostics;
- logs;
- experiment registry.

Ezáltal az AI domain logika nem függ közvetlenül a Streamlittől.

## Composition root

`rag_engine/service.py` a rendszer composition rootja. Itt áll össze a `LabBundle`, majd a kiválasztott stratégia alapján a konkrét RAG pipeline.

```text
config
  ↓
build_lab()
  ├─ ingestion
  ├─ embedder
  ├─ vector store
  ├─ dense/BM25/hybrid retriever
  ├─ LLM provider
  └─ grounded generator
        ↓
create_rag_pipeline()
        ↓
selected strategy
```

## UI boundary

A `ui/` csak presentation layer. A Streamlit oldalak meghívják a `rag_engine` service-eket, de az AI pipeline implementáció nincs a page fájlokba beégetve.

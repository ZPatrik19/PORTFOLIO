# Multi-Index Chunking & Embedding Lab

The project now separates two experiments that are often incorrectly conflated.

## 1. Query-time re-chunking

This is the fast interactive experiment. Retrieval runs against the selected persistent index, then the strongest evidence is reshaped using `fixed`, `recursive`, `structure_aware` or `semantic` chunking before context construction. No embedding matrix is rebuilt. Use this to study **context shape** and final-answer sensitivity.

## 2. Persistent chunk-specific embedding indexes

This is the correct retrieval experiment. Each strategy gets its own chunks and its own embedding matrix under:

```text
01_data/indexes/variants/
├── fixed/
├── recursive/
├── semantic/
└── ...
```

The configured primary index is `structure_aware`, so it can also be selected as the structure-aware experiment without duplicating the large vector matrix.

Current local corpus variants built with 900-character target size and 120-character overlap:

| Strategy | Chunks | Documents | Embedding provider |
|---|---:|---:|---|
| fixed | 21,150 | 43 | LocalHashingEmbeddingProvider |
| recursive | 28,876 | 43 | LocalHashingEmbeddingProvider |
| structure-aware | primary index | 43 | LocalHashingEmbeddingProvider |
| semantic | 67,804 | 43 | LocalHashingEmbeddingProvider |

These local embeddings are deterministic portfolio baselines. For semantic-quality benchmarking, rebuild selected variants with the configured Gemini embedding provider and compare retrieval metrics on the same evaluation set.

## Build variants

```bat
.venv\Scripts\python.exe -m tkip.cli index-variants --strategies fixed recursive semantic
```

or use **Library → Chunking embedding indexes** in the Streamlit UI.

## Why separate indexes?

A vector embedding corresponds to a specific text unit. If chunk boundaries change, both the text and its embedding change. Reusing the old vectors would not test chunking fairly. Therefore the project never mixes vectors from different persistent chunking strategies in one retriever.

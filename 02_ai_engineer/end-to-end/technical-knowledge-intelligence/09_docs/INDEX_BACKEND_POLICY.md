# Index Backend Policy

## Default: NumPy

The local Windows portfolio runtime uses the portable NumPy dense-vector index by default. The project still performs BM25 lexical retrieval, dense cosine retrieval, Reciprocal Rank Fusion, and reranking; choosing NumPy does not remove the hybrid retrieval pipeline.

This default is deliberate because the user library contains more than 20,000 chunks. It avoids using Qdrant embedded/local mode outside its intended small-dataset/testing role and makes first-run setup simpler and deterministic.

## Optional: Qdrant Server

For a production-like deployment, run Qdrant as a normal service (Docker/Qdrant Server) and configure:

```yaml
vector_store:
  provider: qdrant
  qdrant_mode: server
  qdrant_url: http://127.0.0.1:6333
```

`QDRANT_URL` and `QDRANT_API_KEY` environment variables are also supported. The NumPy index is still written as a reproducible fallback.

## Safety fallback

If someone explicitly selects `qdrant_mode: local`, collections larger than `max_local_points` are skipped rather than pushed into embedded Qdrant. The application continues with the NumPy index and records the reason in `01_data/indexes/qdrant_skipped.txt`.

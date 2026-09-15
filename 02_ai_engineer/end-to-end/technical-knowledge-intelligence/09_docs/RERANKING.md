# Reranking

The candidate stage intentionally retrieves more chunks than the answer stage consumes. The default lightweight reranker combines token overlap and hybrid score so the repository runs without downloading a large model. Set `provider: cross_encoder` to use Sentence Transformers CrossEncoder when the environment supports it.

Reranking is benchmarked as an incremental quality/latency tradeoff, not assumed to be beneficial in every corpus.

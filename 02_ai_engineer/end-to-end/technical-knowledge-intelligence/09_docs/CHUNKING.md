# Chunking

Four strategies are implemented: fixed, recursive, structure-aware and lightweight semantic boundary chunking. `structure_aware` is the default because technical books frequently contain headings and code blocks whose integrity matters more than exact token uniformity.

Benchmark dimensions: retrieval Recall@K/MRR, chunk size distribution, duplicate rate, context precision proxy and latency. A larger chunk may increase recall but waste context budget; a smaller chunk can improve precision while losing cross-sentence meaning.

# Retrieval

BM25 covers exact identifiers, API names and uncommon technical terms. Dense retrieval covers semantic paraphrases and Hungarian-to-English matching when a multilingual embedding provider is used. Reciprocal Rank Fusion combines rankings without assuming BM25 and cosine scores share a calibrated scale.

RRF formula: each ranked result contributes `1/(k + rank)`. This is robust for heterogeneous score spaces and easy to audit.

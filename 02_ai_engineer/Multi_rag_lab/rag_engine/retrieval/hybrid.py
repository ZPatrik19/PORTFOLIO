from __future__ import annotations

from rag_engine.retrieval.fusion import reciprocal_rank_fusion, weighted_fusion


class HybridRetriever:
    def __init__(self, dense, sparse, *, fusion: str = "rrf", rrf_k: int = 60, dense_weight: float = 0.5) -> None:
        self.dense, self.sparse = dense, sparse
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight

    def retrieve(self, query: str, top_k: int = 5, candidate_count: int = 20):
        dense = self.dense.retrieve(query, candidate_count)
        sparse = self.sparse.retrieve(query, candidate_count)
        if self.fusion == "weighted":
            return weighted_fusion(dense, sparse, dense_weight=self.dense_weight, top_k=top_k)
        return reciprocal_rank_fusion([dense, sparse], k=self.rrf_k, top_k=top_k)

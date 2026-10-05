from __future__ import annotations


class DenseRetriever:
    def __init__(self, embedder, vector_store) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5):
        return self.vector_store.search(self.embedder.embed_query(query), top_k)

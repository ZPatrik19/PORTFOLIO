import numpy as np

from rag_engine.indexing.hashing import HashingEmbeddingProvider


def test_hashing_embedding_shape_and_normalization():
    provider = HashingEmbeddingProvider(dimensions=64)
    vectors = provider.embed_documents(["hello world", "retrieval augmented generation"])
    assert vectors.shape == (2, 64)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)

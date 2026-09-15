from __future__ import annotations

import numpy as np

from tkip.embeddings import LocalHashingEmbeddingProvider, create_embedding_provider


def test_local_hashing_embeddings_are_deterministic() -> None:
    # Arrange
    provider = LocalHashingEmbeddingProvider(dimension=64)
    text = "retrieval augmented generation"

    # Act
    first = provider.embed_query(text)
    second = provider.embed_query(text)

    # Assert
    np.testing.assert_allclose(first, second)
    assert first.shape == (64,)


def test_local_hashing_embeddings_are_normalized_for_non_empty_text() -> None:
    # Arrange
    provider = LocalHashingEmbeddingProvider(dimension=64)

    # Act
    vector = provider.embed_query("docker kubernetes fastapi")

    # Assert
    assert np.isclose(np.linalg.norm(vector), 1.0)


def test_local_hashing_empty_batch_has_stable_shape() -> None:
    # Arrange
    provider = LocalHashingEmbeddingProvider(dimension=32)

    # Act
    vectors = provider.embed_documents([])

    # Assert
    assert vectors.shape == (0, 32)


def test_embedding_factory_respects_explicit_local_provider(isolated_config: dict) -> None:
    # Arrange
    isolated_config["embedding"]["provider"] = "local_hashing"
    isolated_config["embedding"]["output_dimensionality"] = 48

    # Act
    provider = create_embedding_provider(isolated_config)

    # Assert
    assert isinstance(provider, LocalHashingEmbeddingProvider)
    assert provider.dimension == 48

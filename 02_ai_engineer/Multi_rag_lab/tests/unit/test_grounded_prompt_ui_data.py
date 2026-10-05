from __future__ import annotations

import numpy as np

from rag_engine.generation.grounded import GroundedGenerator
from rag_engine.generation.provider_dummy import DummyLLMProvider
from rag_engine.evaluation.projection import pca_2d


def test_grounded_generator_keeps_prompt_parts() -> None:
    generator = GroundedGenerator(DummyLLMProvider())
    answer, citations = generator.generate("Mi a teszt?", "[S1] Teszt bizonyíték.")
    assert answer
    assert isinstance(citations, list)
    assert "Mi a teszt?" in generator.last_prompt
    assert "[S1] Teszt bizonyíték." in generator.last_prompt
    assert generator.last_prompt_parts["query"] == "Mi a teszt?"
    assert "bizonyíték" in generator.last_prompt_parts["instructions"].lower()


def test_pca_2d_returns_two_dimensions() -> None:
    vectors = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.9, 0.1, 0.4],
            [0.0, 1.0, 0.3],
        ]
    )
    coords = pca_2d(vectors)
    assert coords.shape == (3, 2)
    assert np.isfinite(coords).all()

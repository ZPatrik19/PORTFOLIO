from __future__ import annotations

from pathlib import Path

import pandas as pd
from test_support.factories import make_chunk
from tkip.figures import save_basic_figures


def test_basic_figure_generation_writes_expected_artifacts(tmp_path: Path) -> None:
    # Arrange
    chunks = [
        make_chunk(
            chunk_id="c1",
            title="RAG Guide",
            text="Retrieval augmented generation combines retrieval and generation.",
            keywords=["rag", "retrieval"],
        ),
        make_chunk(
            chunk_id="c2",
            title="RAG Guide",
            text="Reranking improves evidence ordering.",
            keywords=["reranking"],
        ),
    ]
    benchmark = pd.DataFrame(
        [
            {"recall@5": 1.0, "mrr": 1.0, "hit_rate": 1.0, "latency_ms": 2.0},
            {"recall@5": 0.8, "mrr": 0.5, "hit_rate": 1.0, "latency_ms": 3.0},
        ]
    )

    # Act
    save_basic_figures(chunks, pd.DataFrame(), benchmark, tmp_path)

    # Assert
    assert (tmp_path / "chunks_per_document.png").exists()
    assert (tmp_path / "chunk_size_distribution.png").exists()
    assert (tmp_path / "retrieval_metrics.png").exists()
    assert (tmp_path / "retrieval_latency.png").exists()

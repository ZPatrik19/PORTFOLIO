"""Local performance smoke tests.

The threshold is intentionally generous: the goal is to catch accidental severe
algorithmic regressions, not benchmark hardware performance in CI.
"""

from __future__ import annotations

import time

import pytest
from test_support.factories import make_chunk
from tkip.retrieval import BM25Index

pytestmark = pytest.mark.performance


def test_small_bm25_search_stays_within_generous_latency_budget() -> None:
    chunks = [
        make_chunk(
            chunk_id=f"chunk-{index}",
            document_id=f"doc-{index // 10}",
            title=f"Document {index // 10}",
            text=("retrieval augmented generation hybrid search " * 20) + str(index),
        )
        for index in range(200)
    ]
    index = BM25Index(chunks)

    start = time.perf_counter()
    results = index.search("hybrid retrieval", 10)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert len(results) == 10
    assert elapsed_ms < 1_500, f"BM25 local search took {elapsed_ms:.1f} ms, expected < 1500 ms"

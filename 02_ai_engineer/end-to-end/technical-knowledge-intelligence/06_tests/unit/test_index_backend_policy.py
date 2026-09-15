"""Unit tests for vector-store backend safety policy."""

from __future__ import annotations

import pytest
from tkip.indexing import _qdrant_local_is_too_large

pytestmark = pytest.mark.unit


def test_qdrant_local_mode_is_skipped_above_configured_threshold() -> None:
    cfg = {"vector_store": {"qdrant_mode": "local", "max_local_points": 20_000}}

    assert _qdrant_local_is_too_large(cfg, 20_001) is True
    assert _qdrant_local_is_too_large(cfg, 20_000) is False


def test_qdrant_server_mode_ignores_local_collection_threshold() -> None:
    cfg = {"vector_store": {"qdrant_mode": "server", "max_local_points": 1}}

    assert _qdrant_local_is_too_large(cfg, 999_999) is False

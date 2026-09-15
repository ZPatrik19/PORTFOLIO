"""Unit tests for citation validation."""
from __future__ import annotations

import pytest

from test_support.factories import make_chunk, make_citation, make_search_hit
from tkip.citations import validate_citations

pytestmark = pytest.mark.unit


def test_citation_is_valid_when_chunk_exists_in_selected_context() -> None:
    chunk = make_chunk(chunk_id="grounded-chunk")
    citation = make_citation(chunk)
    selected_hits = [make_search_hit(chunk)]

    result = validate_citations([citation], selected_hits)

    assert result["valid"] is True
    assert result.get("errors", []) == []


def test_citation_is_rejected_when_chunk_is_not_in_selected_context() -> None:
    chunk = make_chunk(chunk_id="context-chunk")
    invalid_citation = make_citation(chunk, chunk_id="invented-chunk")
    selected_hits = [make_search_hit(chunk)]

    result = validate_citations([invalid_citation], selected_hits)

    assert result["valid"] is False
    assert result.get("errors")

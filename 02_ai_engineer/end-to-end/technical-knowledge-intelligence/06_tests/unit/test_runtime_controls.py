"""Unit tests for request-time chunking and offline prompt composition."""

from __future__ import annotations

import pytest
from test_support.factories import make_chunk, make_search_hit
from tkip.config import load_config
from tkip.gemini_service import GeminiService
from tkip.runtime_chunking import rechunk_hits

pytestmark = pytest.mark.unit


def test_runtime_rechunking_splits_context_without_changing_document_identity() -> None:
    source_text = (
        "Sentence one about RAG. Sentence two about retrieval. Sentence three about citations. "
    ) * 10
    original_chunk = make_chunk(chunk_id="original", text=source_text, document_id="doc-1")
    original_hit = make_search_hit(original_chunk)

    rechunked_hits, metadata = rechunk_hits(
        [original_hit],
        "recursive",
        260,
        40,
    )

    assert metadata["enabled"] is True
    assert metadata["strategy"] == "recursive"
    assert len(rechunked_hits) > 1
    assert all(hit.chunk.document_id == "doc-1" for hit in rechunked_hits)
    assert all(hit.chunk.chunk_id != "original" for hit in rechunked_hits)


def test_prompt_composition_is_available_without_live_gemini_api() -> None:
    service = GeminiService(load_config(), api_key=None)

    system_prompt, final_prompt = service._compose_prompt(
        "Mi az a RAG?",
        {
            "prompt": "USER INTENT: CONCEPTUAL\nRETRIEVED SOURCES:\n[SOURCE]demo[/SOURCE]",
            "selected_hits": [],
        },
        [],
        "CONCEPTUAL",
        "teacher",
        None,
        "Use a short example.",
        "hu",
    )

    assert "Hungarian" in system_prompt
    assert "Use a short example." in final_prompt
    assert "VALID CITATION OBJECTS" in final_prompt

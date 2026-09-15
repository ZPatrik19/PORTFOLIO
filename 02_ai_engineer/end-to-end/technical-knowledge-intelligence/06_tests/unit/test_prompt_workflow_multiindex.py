"""Unit tests for prompt profiles, workflow definition and index inventory."""
from __future__ import annotations

import pytest

from tkip.config import load_config
from tkip.multi_index import MultiIndexManager
from tkip.presets import ANSWER_PRESETS, CHUNK_PRESETS
from tkip.prompt_engineering import PROFILES, PROFILE_KEYS, local_optimize
from tkip.workflow_graph import workflow_dot, workflow_rows

pytestmark = pytest.mark.unit


def test_all_prompt_profiles_preserve_original_question() -> None:
    question = "How does RAG work?"

    assert len(PROFILE_KEYS) == 16
    for profile_key in PROFILE_KEYS:
        optimized = local_optimize(question, profile_key)
        assert question in optimized["optimized"]
        assert optimized["changed"] is True
        assert profile_key in PROFILES


def test_numbered_workflow_contains_application_and_ai_layers() -> None:
    rows = workflow_rows("full")
    labels = {row["label"] for row in rows}

    assert "Streamlit research workspace" in labels
    assert "FastAPI service boundary" in labels
    assert "Advanced prompt optimizer" in labels
    assert "BM25 + Dense retrieval" in labels
    assert "Citation & output validation" in labels

    dot = workflow_dot("full")
    assert "digraph" in dot
    assert "bm25" in dot
    assert "dense" in dot
    assert "chunk_compact" in dot
    assert "chunk_semantic" in dot


def test_multi_index_inventory_exposes_all_supported_variants() -> None:
    manager = MultiIndexManager(load_config())
    available_names = {item["name"] for item in manager.available()}

    expected = {"primary", "fixed", "recursive", "structure_aware", "semantic"}
    assert expected.issubset(available_names)


def test_answer_and_chunk_presets_define_clear_quality_levels() -> None:
    assert set(ANSWER_PRESETS) == {"economy", "recommended", "deep", "max_quality"}
    assert set(CHUNK_PRESETS) == {"compact", "balanced", "semantic_deep"}
    assert ANSWER_PRESETS["recommended"]["max_output_tokens"] < ANSWER_PRESETS["max_quality"]["max_output_tokens"]

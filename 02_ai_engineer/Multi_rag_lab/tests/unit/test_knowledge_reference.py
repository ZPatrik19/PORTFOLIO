from rag_engine.presets import CHUNKING_STRATEGIES, RAG_STRATEGIES
from rag_engine.knowledge import (
    CATEGORY_ORDER,
    all_reference_entries,
    filter_reference_entries,
    reference_markdown,
    reference_stats,
)


def test_reference_keys_are_unique_and_categories_valid():
    entries = all_reference_entries()
    assert entries
    keys = [item.key for item in entries]
    assert len(keys) == len(set(keys))
    assert all(item.category in CATEGORY_ORDER for item in entries)


def test_reference_contains_project_strategies():
    entries = all_reference_entries()
    keys = {item.key for item in entries}
    assert {f"chunking_{key}" for key in CHUNKING_STRATEGIES}.issubset(keys)
    assert {f"rag_{key}" for key in RAG_STRATEGIES}.issubset(keys)


def test_reference_has_substantial_metric_and_formula_coverage():
    stats = reference_stats(all_reference_entries())
    assert stats["entries"] >= 70
    assert stats["metrics"] >= 25
    assert stats["formulas"] >= 12
    assert stats["strategies"] >= len(CHUNKING_STRATEGIES) + len(RAG_STRATEGIES)


def test_reference_search_finds_mrr_and_cuda():
    entries = all_reference_entries()
    mrr = filter_reference_entries(entries, query="MRR")
    cuda = filter_reference_entries(entries, query="CUDA")
    assert any(item.key == "mrr" for item in mrr)
    assert any(item.key == "cuda" for item in cuda)


def test_reference_markdown_export_contains_sections():
    markdown = reference_markdown(all_reference_entries())
    assert "# Multi-RAG Engineering Lab – Fogalomtár és metrikák" in markdown
    assert "## Retrieval metrikák" in markdown
    assert "### Recall@K" in markdown
    assert "### TTFT – Time to First Token" in markdown

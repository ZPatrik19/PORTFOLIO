"""Static regression contracts for the portfolio UI release."""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _questions() -> dict[str, list[str]]:
    source = (PROJECT_ROOT / "05_ui" / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "QUESTIONS":
                    return ast.literal_eval(node.value)
    raise AssertionError("QUESTIONS assignment not found")


def test_workflow_page_is_single_interactive_orchestration_lifecycle() -> None:
    pages = (PROJECT_ROOT / "05_ui" / "ui_pages.py").read_text(encoding="utf-8")
    graph = (PROJECT_ROOT / "05_ui" / "ui_workflow.py").read_text(encoding="utf-8")

    assert "render_orchestration_graph" in pages
    assert "go.Figure" in graph
    assert "RunnableBranch" in graph
    assert "BM25" in graph and "Dense" in graph
    assert "ToolNode" in graph
    assert "Human-in-the-loop" in graph
    assert "feedback loop" in graph.lower()
    assert "scrollZoom" in graph
    assert "st.tabs(" not in pages[pages.index("def render_workflow_page") :]


def test_benchmark_ui_has_preloaded_snapshot_and_metric_guides() -> None:
    source = (PROJECT_ROOT / "05_ui" / "ui_experiments.py").read_text(encoding="utf-8")

    assert "_render_benchmark_snapshot(ui_lang)" in source
    assert source.count("_render_metric_guide(") >= 6
    assert "retrieval_methods_summary.csv" in source
    assert "generation_deterministic_metrics.csv" in source
    assert "Példa eszközhívási JSON" in source
    assert "prompt-transzformáció JSON" in source


def test_question_bank_is_substantially_expanded_and_bilingual() -> None:
    questions = _questions()

    assert len(questions["hu"]) >= 40
    assert len(questions["en"]) >= 40
    assert len(questions["hu"]) == len(questions["en"])


def test_lightweight_demo_corpus_contains_many_documents_without_user_library_files() -> None:
    reference_docs = list((PROJECT_ROOT / "01_data" / "reference_docs").glob("demo_*.md"))
    user_docs = [
        p
        for p in (PROJECT_ROOT / "01_data" / "user_library").glob("*")
        if p.is_file() and p.name != ".gitkeep"
    ]

    assert len(reference_docs) >= 30
    assert user_docs == []


def test_launcher_restarts_stale_tki_ports_by_default() -> None:
    source = (PROJECT_ROOT / "05_scripts" / "launch_app.py").read_text(encoding="utf-8")

    assert "restart_existing: bool = True" in source
    assert "_stop_port_owner(API_PORT" in source
    assert "_stop_port_owner(UI_PORT" in source
    assert "--reuse-existing" in source

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "ui" / "components" / "common.py"
APP = ROOT / "ui" / "app.py"
OVERVIEW = ROOT / "ui" / "pages" / "overview.py"
GUIDE = ROOT / "ui" / "pages" / "guide.py"
CHARTS = ROOT / "ui" / "components" / "charts.py"


def test_context_profile_uses_callback_instead_of_post_widget_state_mutation() -> None:
    source = COMMON.read_text(encoding="utf-8")
    assert "def _sync_context_budget_from_profile" in source
    assert "on_change=_sync_context_budget_from_profile" in source
    assert "Ajánlott tokenkeret alkalmazása" not in source


def test_obsolete_end_to_end_workflow_diagram_is_removed() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in [APP, OVERVIEW, GUIDE, CHARTS])
    assert "render_workflow_panel" not in combined
    assert "architecture_overview_chart" not in combined
    assert "workflow_pipeline_chart" not in combined
    assert "Aktív end-to-end pipeline" not in combined
    assert not (ROOT / "ui" / "components" / "workflow.py").exists()


def test_runtime_page_does_not_directly_mutate_sidebar_widget_keys() -> None:
    runtime = (ROOT / "ui" / "pages" / "runtime_lab.py").read_text(encoding="utf-8")
    forbidden = {
        "llm_provider",
        "ollama_profile",
        "faiss_profile",
        "cuda_profile",
        "embedding_device",
        "reranker_device",
        "vector_device",
    }
    for key in forbidden:
        assert f'st.session_state["{key}"] =' not in runtime
    assert "queue_widget_state_updates" in runtime
    assert "st.rerun()" in runtime

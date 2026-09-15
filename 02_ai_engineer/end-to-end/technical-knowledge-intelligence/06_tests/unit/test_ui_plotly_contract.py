from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_plotly_is_runtime_dependency() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert '"plotly>=6.5"' in pyproject


def test_benchmark_ui_uses_plotly_instead_of_static_charts() -> None:
    source = (ROOT / "05_ui" / "ui_experiments.py").read_text(encoding="utf-8")
    assert "st.plotly_chart" in source
    assert "st.bar_chart" not in source
    assert "st.line_chart" not in source
    assert "_render_image_gallery" not in source


def test_workflow_ui_is_interactive_plotly_graph() -> None:
    workflow = (ROOT / "05_ui" / "ui_workflow.py").read_text(encoding="utf-8")
    pages = (ROOT / "05_ui" / "ui_pages.py").read_text(encoding="utf-8")
    assert "go.Figure" in workflow
    assert "scrollZoom" in workflow
    assert "RunnableBranch" in workflow
    assert "BM25" in workflow and "Dense" in workflow
    assert "Human-in-the-loop" in workflow
    assert "render_orchestration_graph" in pages


def test_all_plotly_charts_have_unique_explicit_keys() -> None:
    import ast

    keys: list[str] = []
    for relative in [Path("05_ui/ui_experiments.py"), Path("05_ui/ui_workflow.py")]:
        source_path = ROOT / relative
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "plotly_chart"):
                continue
            key_kw = next((kw for kw in node.keywords if kw.arg == "key"), None)
            assert key_kw is not None, f"Missing Plotly key in {relative}:{node.lineno}"
            assert isinstance(key_kw.value, ast.Constant) and isinstance(key_kw.value.value, str), (
                f"Plotly key must be a literal string in {relative}:{node.lineno}"
            )
            keys.append(key_kw.value.value)

    assert keys
    assert len(keys) == len(set(keys)), f"Duplicate Plotly keys found: {keys}"


def test_workflow_uses_multiple_node_shapes_and_restrained_phase_colors() -> None:
    source = (ROOT / "05_ui" / "ui_workflow.py").read_text(encoding="utf-8")
    for symbol in ["square", "hexagon", "diamond", "pentagon", "circle", "bowtie", "star"]:
        assert f'symbol="{symbol}"' in source
    for accent in ["#2dd4bf", "#60a5fa", "#818cf8", "#f59e0b", "#4ade80"]:
        assert accent in source

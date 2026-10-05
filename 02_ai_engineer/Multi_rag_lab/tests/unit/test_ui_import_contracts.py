from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHARTS_PATH = ROOT / "ui" / "components" / "charts.py"


def _defined_symbols(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def test_all_ui_component_chart_imports_exist() -> None:
    """Catch renamed/missing chart imports before Streamlit startup."""
    chart_symbols = _defined_symbols(CHARTS_PATH)
    missing: list[str] = []

    for path in (ROOT / "ui").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "ui.components.charts":
                for imported in node.names:
                    if imported.name not in chart_symbols:
                        relative = path.relative_to(ROOT)
                        missing.append(f"{relative}: {imported.name}")

    assert not missing, "Missing ui.components.charts exports: " + ", ".join(missing)

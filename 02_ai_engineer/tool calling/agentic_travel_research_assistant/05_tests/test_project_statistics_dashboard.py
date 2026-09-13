from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "08_ui"
if str(UI) not in sys.path:
    sys.path.insert(0, str(UI))
MODULE_PATH = UI / "project_statistics_dashboard.py"
spec = importlib.util.spec_from_file_location("project_statistics_dashboard", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_dataset_scale_figure_has_trace():
    df = pd.DataFrame([
        {"dataset": "hotels.csv", "rows": 180000, "columns": 10, "missing_cells": 0, "exact_duplicate_rows": 0, "memory_mb": 50.0},
        {"dataset": "restaurants.csv", "rows": 90000, "columns": 8, "missing_cells": 0, "exact_duplicate_rows": 0, "memory_mb": 30.0},
    ])
    fig = module.build_dataset_scale_chart(df, "rows")
    assert len(fig.data) >= 1
    assert fig.layout.title.text == "Dataset scale"


def test_methodology_radar_contains_each_methodology():
    df = pd.DataFrame([
        {"methodology": "rule_based", "tool_selection_accuracy": .2, "tool_f1": .4, "argument_accuracy": .3, "task_success": .1},
        {"methodology": "plan_execute", "tool_selection_accuracy": .6, "tool_f1": .8, "argument_accuracy": .7, "task_success": .5},
        {"methodology": "ml_router", "tool_selection_accuracy": .9, "tool_f1": .97, "argument_accuracy": .95, "task_success": .7},
    ])
    fig = module.build_methodology_radar(df)
    assert len(fig.data) == 3
    assert {trace.name for trace in fig.data} == {"Rule-based baseline", "Plan → Execute", "ML Router"}


def test_router_metrics_is_grouped_precision_recall_f1():
    df = pd.DataFrame([
        {"label": "hotel", "precision": .95, "recall": .9, "f1": .925, "support": 100},
        {"label": "weather", "precision": .9, "recall": .85, "f1": .875, "support": 100},
    ])
    fig = module.build_router_metrics_chart(df)
    assert len(fig.data) == 3
    assert {trace.name for trace in fig.data} == {"Precision", "Recall", "F1"}


def test_price_histogram_contains_histogram_and_box():
    fig = module.build_price_histogram(pd.Series([10, 20, 30, 40, 50]), "Prices", "EUR", bins=5)
    types = {trace.type for trace in fig.data}
    assert "histogram" in types
    assert "box" in types


def test_all_project_statistics_plotly_calls_have_unique_keys():
    import ast

    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    keys = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "plot"):
            continue
        key_kw = next((kw for kw in node.keywords if kw.arg == "key"), None)
        assert key_kw is not None, "Every dashboard plot() call must define an explicit unique key."
        assert isinstance(key_kw.value, ast.Constant) and isinstance(key_kw.value.value, str)
        keys.append(key_kw.value.value)
    assert keys
    assert len(keys) == len(set(keys)), f"Duplicate Plotly keys found: {keys}"


def test_ui_uses_current_streamlit_width_api():
    ui_dir = ROOT / "08_ui"
    offenders = []
    for path in ui_dir.rglob("*.py"):
        if "use_container_width" in path.read_text(encoding="utf-8"):
            offenders.append(path.name)
    assert offenders == [], f"Deprecated use_container_width remains in: {offenders}"

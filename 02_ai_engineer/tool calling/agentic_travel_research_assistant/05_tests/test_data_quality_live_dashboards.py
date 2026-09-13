from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "08_ui"
if str(UI) not in sys.path:
    sys.path.insert(0, str(UI))


def _load(name: str):
    path = UI / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


dq = _load("data_quality_dashboard.py")
live = _load("live_statistics_dashboard.py")


def test_data_quality_gauge_and_leakage_heatmap_build():
    gauge = dq.build_quality_gauge(1.0)
    assert len(gauge.data) == 1
    assert gauge.data[0].type == "indicator"

    leakage = {
        "train__validation": {"shared_patterns": 0, "overlap_over_smaller_split": 0.0},
        "test__train": {"shared_patterns": 0, "overlap_over_smaller_split": 0.0},
        "test__validation": {"shared_patterns": 0, "overlap_over_smaller_split": 0.0},
    }
    fig = dq.build_leakage_heatmap(leakage)
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"


def test_query_and_entity_quality_figures_build():
    qdf = pd.DataFrame([
        {"dataset": "Router", "rows": 240000, "unique_queries": 240000, "normalized_patterns": 180000,
         "pattern_ratio": .75, "largest_pattern_share": .001, "exact_duplicate_rate": 0.0, "median_pattern_frequency": 1.0},
        {"dataset": "Challenge", "rows": 36000, "unique_queries": 36000, "normalized_patterns": 20000,
         "pattern_ratio": .56, "largest_pattern_share": .003, "exact_duplicate_rate": 0.0, "median_pattern_frequency": 1.0},
    ])
    assert len(dq.build_query_diversity_scatter(qdf).data) >= 1
    assert len(dq.build_query_risk_chart(qdf).data) >= 1

    edf = pd.DataFrame([
        {"dataset": "Hotels", "rows": 180000, "unique_names": 167000, "name_unique_ratio": .93,
         "name_skeleton_ratio": .42, "largest_skeleton_share": .01, "duplicate_name_rate": .07},
        {"dataset": "Restaurants", "rows": 90000, "unique_names": 90000, "name_unique_ratio": 1.0,
         "name_skeleton_ratio": .51, "largest_skeleton_share": .01, "duplicate_name_rate": 0.0},
    ])
    assert len(dq.build_entity_diversity_chart(edf).data) >= 1
    assert len(dq.build_entity_risk_chart(edf).data) >= 1


def test_live_usage_figures_build():
    daily = pd.DataFrame([
        {"date": "2026-09-12", "questions": 4, "tool_calls": 9, "avg_latency_ms": 20.0, "success_rate": .75},
        {"date": "2026-09-13", "questions": 8, "tool_calls": 21, "avg_latency_ms": 18.0, "success_rate": 1.0},
    ])
    assert len(live.build_daily_volume_chart(daily).data) == 2
    assert len(live.build_success_latency_chart(daily).data) == 2

    tools = pd.DataFrame([
        {"tool_name": "get_weather", "calls": 10, "successful_calls": 9, "failed_calls": 1, "avg_latency_ms": 8.0, "min_latency_ms": 4.0, "max_latency_ms": 14.0},
        {"tool_name": "search_hotels", "calls": 6, "successful_calls": 6, "failed_calls": 0, "avg_latency_ms": 11.0, "min_latency_ms": 6.0, "max_latency_ms": 19.0},
    ])
    assert len(live.build_tool_usage_chart(tools).data) >= 1


def test_all_ui_plotly_calls_have_explicit_globally_unique_keys():
    keys: list[str] = []
    for path in UI.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Name) and func.id == "plot"):
                continue
            key_kw = next((kw for kw in node.keywords if kw.arg == "key"), None)
            assert key_kw is not None, f"Missing Plotly key in {path.name}:{node.lineno}"
            assert isinstance(key_kw.value, ast.Constant) and isinstance(key_kw.value.value, str)
            keys.append(key_kw.value.value)
    assert keys
    assert len(keys) == len(set(keys)), f"Duplicate Plotly keys found across UI: {keys}"

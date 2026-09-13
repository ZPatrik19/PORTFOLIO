from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "08_ui"
if str(UI) not in sys.path:
    sys.path.insert(0, str(UI))

from chart_theme import CHART_HEIGHT, normalize_chart_theme, style_figure


def test_light_theme_uses_black_text_on_white_background():
    fig = style_figure(go.Figure(go.Bar(x=["A"], y=[1])), "light")
    assert fig.layout.paper_bgcolor == "#FFFFFF"
    assert fig.layout.plot_bgcolor == "#FFFFFF"
    assert fig.layout.font.color == "#111111"
    assert fig.layout.xaxis.tickfont.color == "#111111"
    assert fig.layout.yaxis.tickfont.color == "#111111"
    assert fig.layout.height == CHART_HEIGHT


def test_dark_theme_uses_white_text_on_dark_background():
    fig = style_figure(go.Figure(go.Bar(x=["A"], y=[1])), "dark")
    assert fig.layout.paper_bgcolor == "#0E1117"
    assert fig.layout.plot_bgcolor == "#0E1117"
    assert fig.layout.font.color == "#FFFFFF"
    assert fig.layout.xaxis.tickfont.color == "#FFFFFF"
    assert fig.layout.yaxis.tickfont.color == "#FFFFFF"
    assert fig.layout.height == CHART_HEIGHT


def test_unknown_theme_falls_back_to_light():
    assert normalize_chart_theme("something") == "light"


def test_all_dashboard_renderers_accept_chart_theme():
    for name, func_name in [
        ("project_statistics_dashboard.py", "render_dashboard"),
        ("data_quality_dashboard.py", "render_data_quality_dashboard"),
        ("live_statistics_dashboard.py", "render_live_statistics_dashboard"),
    ]:
        spec = importlib.util.spec_from_file_location(name.replace(".py", ""), UI / name)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        import inspect
        assert "chart_theme" in inspect.signature(getattr(module, func_name)).parameters

from __future__ import annotations

import sys
import types
from pathlib import Path

import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The unit-test container does not install Streamlit; charts only need it at render time.
sys.modules.setdefault("streamlit", types.ModuleType("streamlit"))

from ui.components.charts import _apply_layout


def test_apply_layout_preserves_existing_title_without_duplicate_keyword() -> None:
    fig = go.Figure()
    fig.update_layout(title="Meglévő cím")

    _apply_layout(fig, height=321)

    assert fig.layout.title.text == "Meglévő cím"
    assert fig.layout.height == 321


def test_apply_layout_can_override_title_explicitly() -> None:
    fig = go.Figure()
    fig.update_layout(title="Régi cím")

    _apply_layout(fig, title="Új cím", height=222)

    assert fig.layout.title.text == "Új cím"
    assert fig.layout.height == 222

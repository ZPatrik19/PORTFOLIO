"""Shared high-contrast theming helpers for all analytics dashboards.

The Streamlit application exposes a user-selectable light/dark analytics theme.
This module applies the same typography, background, grid, legend, hover and
fixed chart-height rules to every Plotly figure so Project Statistics, Data
Quality and Live Statistics remain visually consistent.
"""
from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

CHART_HEIGHT = 460

THEMES: dict[str, dict[str, str]] = {
    "light": {
        "paper": "#FFFFFF",
        "plot": "#FFFFFF",
        "text": "#111111",
        "muted": "#262626",
        "grid": "#D1D5DB",
        "axis": "#4B5563",
        "zero": "#9CA3AF",
        "border": "#CBD5E1",
        "hover_bg": "#FFFFFF",
        "hover_text": "#111111",
    },
    "dark": {
        "paper": "#0E1117",
        "plot": "#0E1117",
        "text": "#FFFFFF",
        "muted": "#F3F4F6",
        "grid": "#374151",
        "axis": "#D1D5DB",
        "zero": "#6B7280",
        "border": "#4B5563",
        "hover_bg": "#111827",
        "hover_text": "#FFFFFF",
    },
}


def normalize_chart_theme(theme: str | None) -> str:
    value = str(theme or "light").strip().lower()
    return value if value in THEMES else "light"


def style_figure(fig: go.Figure, theme: str = "light", *, height: int = CHART_HEIGHT) -> go.Figure:
    """Apply a high-contrast theme and fixed height to a Plotly figure."""
    mode = normalize_chart_theme(theme)
    colors = THEMES[mode]

    fig.update_layout(
        height=height,
        paper_bgcolor=colors["paper"],
        plot_bgcolor=colors["plot"],
        font=dict(color=colors["text"], size=13),
        title=dict(font=dict(color=colors["text"], size=20)),
        legend=dict(
            font=dict(color=colors["text"]),
            title=dict(font=dict(color=colors["text"])),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor=colors["hover_bg"],
            bordercolor=colors["border"],
            font=dict(color=colors["hover_text"], size=13),
        ),
        margin=dict(l=24, r=24, t=72, b=36),
    )

    axis_style: dict[str, Any] = {
        "gridcolor": colors["grid"],
        "zerolinecolor": colors["zero"],
        "linecolor": colors["axis"],
        "tickfont": dict(color=colors["text"]),
        "title_font": dict(color=colors["text"]),
        "showline": True,
        "mirror": False,
    }
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)

    # Existing annotations include donut-centre totals and dashboard call-outs.
    if fig.layout.annotations:
        for annotation in fig.layout.annotations:
            font = dict(annotation.font.to_plotly_json()) if annotation.font else {}
            font["color"] = colors["text"]
            annotation.font = font

    # Keep polar/radar labels readable too.
    if getattr(fig.layout, "polar", None):
        fig.update_layout(
            polar=dict(
                bgcolor=colors["plot"],
                angularaxis=dict(
                    color=colors["text"],
                    gridcolor=colors["grid"],
                    linecolor=colors["axis"],
                    tickfont=dict(color=colors["text"]),
                ),
                radialaxis=dict(
                    color=colors["text"],
                    gridcolor=colors["grid"],
                    linecolor=colors["axis"],
                    tickfont=dict(color=colors["text"]),
                ),
            )
        )

    # Outside data labels on Cartesian traces should follow the selected
    # foreground color. Pie/donut labels are intentionally left to Plotly so
    # they can contrast against each slice rather than the page background.
    for trace in fig.data:
        if getattr(trace, "type", "") in {"bar", "scatter", "scattergl", "scatterpolar"} and hasattr(trace, "textfont"):
            try:
                trace.textfont.color = colors["text"]
            except Exception:
                pass

    # Layout-level continuous color axes are used by Plotly Express.
    for layout_key in fig.layout:
        if not str(layout_key).startswith("coloraxis"):
            continue
        axis = getattr(fig.layout, layout_key, None)
        if axis is None or getattr(axis, "colorbar", None) is None:
            continue
        axis.colorbar.tickfont = dict(color=colors["text"])
        if getattr(axis.colorbar, "title", None) is not None:
            axis.colorbar.title.font = dict(color=colors["text"])

    # Colorbar labels can otherwise inherit low-contrast template defaults.
    for trace in fig.data:
        colorbars = []
        direct = getattr(trace, "colorbar", None)
        if direct is not None:
            colorbars.append(direct)
        marker = getattr(trace, "marker", None)
        marker_bar = getattr(marker, "colorbar", None) if marker is not None else None
        if marker_bar is not None:
            colorbars.append(marker_bar)
        for colorbar in colorbars:
            colorbar.tickfont = dict(color=colors["text"])
            if getattr(colorbar, "title", None) is not None:
                colorbar.title.font = dict(color=colors["text"])

    return fig


def render_plotly(st, fig: go.Figure, *, key: str, theme: str = "light", config: dict | None = None, height: int = CHART_HEIGHT) -> None:
    """Style and render a Plotly figure with a unique Streamlit key."""
    merged = {"displaylogo": False, "responsive": True}
    if config:
        merged.update(config)
    st.plotly_chart(
        style_figure(fig, theme, height=height),
        width="stretch",
        key=key,
        config=merged,
        theme=None,
    )

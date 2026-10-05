from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ui.components.tokens import CHART_PALETTE


_BASE_LAYOUT = dict(
    height=460,
    margin=dict(l=24, r=24, t=76, b=108),
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0, font=dict(size=10)),
)


def performance_latency_chart(frame: pd.DataFrame, *, key: str) -> None:
    if frame.empty:
        return
    cols = [c for c in ["Komponens", "Átlag ms", "P95 ms"] if c in frame.columns]
    if len(cols) < 3:
        return
    long = frame[cols].melt(id_vars="Komponens", var_name="Metrika", value_name="Idő ms")
    fig = px.bar(
        long,
        x="Idő ms",
        y="Komponens",
        color="Metrika",
        barmode="group",
        orientation="h",
        title="Komponensenkénti válaszidő · átlag és P95",
        color_discrete_sequence=CHART_PALETTE,
    )
    fig.update_yaxes(automargin=True, title=None)
    fig.update_layout(**_BASE_LAYOUT)
    st.plotly_chart(fig, width="stretch", key=key)


def performance_throughput_chart(frame: pd.DataFrame, *, key: str) -> None:
    if frame.empty or "Áteresztőképesség/s" not in frame.columns:
        return
    fig = px.bar(
        frame,
        x="Áteresztőképesség/s",
        y="Komponens",
        color="Eszköz" if "Eszköz" in frame.columns else None,
        orientation="h",
        title="Komponensenkénti áteresztőképesség",
        color_discrete_sequence=CHART_PALETTE,
    )
    fig.update_yaxes(automargin=True, title=None)
    fig.update_layout(**_BASE_LAYOUT)
    st.plotly_chart(fig, width="stretch", key=key)


def resource_usage_chart(frame: pd.DataFrame, *, key: str) -> None:
    if frame.empty:
        return
    cols = [c for c in ["Komponens", "RAM MB", "Peak VRAM MB"] if c in frame.columns]
    if len(cols) < 2:
        return
    long = (
        frame[cols]
        .melt(id_vars="Komponens", var_name="Erőforrás", value_name="Memória MB")
        .dropna(subset=["Memória MB"])
    )
    if long.empty:
        return
    fig = px.bar(
        long,
        x="Memória MB",
        y="Komponens",
        color="Erőforrás",
        barmode="group",
        orientation="h",
        title="RAM / VRAM erőforrásprofil",
        color_discrete_sequence=CHART_PALETTE,
    )
    fig.update_yaxes(automargin=True, title=None)
    fig.update_layout(**_BASE_LAYOUT)
    st.plotly_chart(fig, width="stretch", key=key)


def device_speedup_chart(frame: pd.DataFrame, *, key: str) -> None:
    """CPU/CUDA speedup for components measured on both devices."""
    if frame.empty or not {"Komponens", "Eszköz", "Összes ms"}.issubset(frame.columns):
        return
    rows: list[dict[str, object]] = []
    for component, group in frame.groupby("Komponens"):
        by_device = {str(row["Eszköz"]).lower(): row for _, row in group.iterrows()}
        cpu = by_device.get("cpu")
        cuda = by_device.get("cuda")
        if cpu is None or cuda is None:
            continue
        cpu_ms = float(cpu.get("Összes ms") or 0.0)
        cuda_ms = float(cuda.get("Összes ms") or 0.0)
        if cpu_ms > 0 and cuda_ms > 0:
            rows.append({"Komponens": component, "Gyorsulás": cpu_ms / cuda_ms})
    if not rows:
        st.info("Nincs olyan komponens, amelyet ugyanebben a futásban CPU-n és CUDA-n is megmértünk.")
        return
    speedup = pd.DataFrame(rows).sort_values("Gyorsulás", ascending=True)
    fig = px.bar(
        speedup,
        x="Gyorsulás",
        y="Komponens",
        orientation="h",
        text="Gyorsulás",
        title="CUDA gyorsulás a CPU-hoz képest",
        color_discrete_sequence=CHART_PALETTE,
    )
    fig.update_traces(texttemplate="%{x:.2f}×", textposition="outside")
    fig.add_vline(x=1.0, line_dash="dash", opacity=0.55, annotation_text="1×")
    fig.update_yaxes(automargin=True, title=None)
    fig.update_layout(**_BASE_LAYOUT, showlegend=False)
    st.plotly_chart(fig, width="stretch", key=key)


def performance_total_time_chart(frame: pd.DataFrame, *, key: str) -> None:
    if frame.empty or not {"Komponens", "Összes ms"}.issubset(frame.columns):
        return
    usable = frame.dropna(subset=["Összes ms"]).sort_values("Összes ms", ascending=True)
    if usable.empty:
        return
    fig = px.bar(
        usable,
        x="Összes ms",
        y="Komponens",
        color="Eszköz" if "Eszköz" in usable.columns else None,
        orientation="h",
        title="Teljes benchmark-idő komponensenként",
        color_discrete_sequence=CHART_PALETTE,
    )
    fig.update_yaxes(automargin=True, title=None)
    fig.update_layout(**_BASE_LAYOUT)
    st.plotly_chart(fig, width="stretch", key=key)


def llm_telemetry_chart(frame: pd.DataFrame, *, key: str) -> None:
    if frame.empty:
        return
    cols = [c for c in ["Komponens", "TTFT ms", "Token/s", "Kimeneti token", "Kontextus token"] if c in frame.columns]
    if len(cols) < 2:
        return
    long = frame[cols].melt(id_vars="Komponens", var_name="Metrika", value_name="Érték").dropna(subset=["Érték"])
    if long.empty:
        return
    fig = px.bar(
        long,
        x="Komponens",
        y="Érték",
        color="Metrika",
        barmode="group",
        title="LLM és end-to-end telemetria",
        color_discrete_sequence=CHART_PALETTE,
    )
    fig.update_layout(**_BASE_LAYOUT)
    st.plotly_chart(fig, width="stretch", key=key)

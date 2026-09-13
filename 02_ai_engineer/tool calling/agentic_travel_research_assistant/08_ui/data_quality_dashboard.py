"""Interactive Plotly dashboard for the Data Quality UI.

The dashboard consumes the persisted quality audit JSON produced by
``travel_agent.quality.audit_all``. Figure construction is kept separate from
Streamlit rendering so visual logic can be unit-tested without launching the
web application.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from chart_theme import render_plotly

PALETTE = ["#2563EB", "#14B8A6", "#8B5CF6", "#F59E0B", "#EF4444", "#06B6D4", "#84CC16", "#EC4899"]
PASS = "#16A34A"
FAIL = "#DC2626"
WARN = "#F59E0B"
BG = "#F8FAFC"
GRID = "#E2E8F0"


def _human(name: str) -> str:
    return str(name).replace(".csv", "").replace("_", " ").title()


def _layout(fig: go.Figure, *, height: int = 440, legend_horizontal: bool = False) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=18, r=18, t=72, b=28),
        hoverlabel=dict(font_size=13),
        font=dict(size=13),
        paper_bgcolor="white",
        plot_bgcolor="white",
        title=dict(x=0.01, xanchor="left", font=dict(size=20)),
        legend=dict(title_text=""),
    )
    if legend_horizontal:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def build_quality_gauge(pass_rate: float) -> go.Figure:
    pct = max(0.0, min(1.0, float(pass_rate))) * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 38}},
        title={"text": "Quality gate pass rate", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": PASS if pct >= 99 else WARN if pct >= 80 else FAIL, "thickness": 0.28},
            "bgcolor": "#F1F5F9",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 70], "color": "#FEE2E2"},
                {"range": [70, 90], "color": "#FEF3C7"},
                {"range": [90, 100], "color": "#DCFCE7"},
            ],
            "threshold": {"line": {"color": "#0F172A", "width": 3}, "thickness": 0.8, "value": 100},
        },
    ))
    return _layout(fig, height=330)


def build_gate_status_chart(gates: dict[str, bool]) -> go.Figure:
    rows = [{"gate": _human(k), "score": 1 if v else 0, "status": "Pass" if v else "Fail"} for k, v in gates.items()]
    df = pd.DataFrame(rows).sort_values(["score", "gate"], ascending=[True, True])
    fig = px.bar(
        df,
        x="score",
        y="gate",
        orientation="h",
        color="status",
        color_discrete_map={"Pass": PASS, "Fail": FAIL},
        text="status",
        title="Quality gates",
        labels={"score": "Status", "gate": "Gate"},
        hover_data={"score": False},
    )
    fig.update_traces(textposition="inside", insidetextanchor="middle")
    fig.update_xaxes(range=[0, 1], showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(title="")
    return _layout(fig, height=max(380, len(df) * 50 + 110))


def query_metrics_frame(report: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, m in report.get("query_datasets", {}).items():
        n = max(1, int(m.get("rows", 0)))
        rows.append({
            "dataset": _human(name),
            "rows": int(m.get("rows", 0)),
            "unique_queries": int(m.get("unique_queries", 0)),
            "normalized_patterns": int(m.get("normalized_patterns", 0)),
            "pattern_ratio": float(m.get("normalized_pattern_ratio", 0)),
            "largest_pattern_share": float(m.get("largest_pattern_share", 0)),
            "exact_duplicate_rate": float(m.get("exact_duplicate_rows", 0)) / n,
            "median_pattern_frequency": float(m.get("median_pattern_frequency", 0)),
        })
    return pd.DataFrame(rows)


def entity_metrics_frame(report: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, m in report.get("entity_datasets", {}).items():
        n = max(1, int(m.get("rows", 0)))
        rows.append({
            "dataset": _human(name),
            "rows": int(m.get("rows", 0)),
            "unique_names": int(m.get("unique_names", 0)),
            "name_unique_ratio": float(m.get("name_unique_ratio", 0)),
            "name_skeleton_ratio": float(m.get("name_skeleton_ratio", 0)),
            "largest_skeleton_share": float(m.get("largest_name_skeleton_share", 0)),
            "duplicate_name_rate": float(m.get("duplicate_names", 0)) / n,
        })
    return pd.DataFrame(rows)


def build_query_diversity_scatter(df: pd.DataFrame) -> go.Figure:
    work = df.copy()
    fig = px.scatter(
        work,
        x="rows",
        y="pattern_ratio",
        size="normalized_patterns",
        color="dataset",
        color_discrete_sequence=PALETTE,
        text="dataset",
        size_max=42,
        title="Query corpus diversity vs scale",
        labels={"rows": "Rows", "pattern_ratio": "Normalized pattern ratio"},
        hover_data={
            "unique_queries": ":,",
            "normalized_patterns": ":,",
            "largest_pattern_share": ":.2%",
            "exact_duplicate_rate": ":.2%",
            "median_pattern_frequency": ":.1f",
        },
    )
    fig.update_traces(textposition="top center")
    fig.update_xaxes(type="log", tickformat="~s", gridcolor=GRID)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05], gridcolor=GRID)
    fig.update_layout(showlegend=False)
    return _layout(fig, height=470)


def build_query_risk_chart(df: pd.DataFrame) -> go.Figure:
    melted = df.melt(
        id_vars="dataset",
        value_vars=["largest_pattern_share", "exact_duplicate_rate"],
        var_name="risk",
        value_name="value",
    )
    melted["risk"] = melted["risk"].map({
        "largest_pattern_share": "Largest pattern share",
        "exact_duplicate_rate": "Exact duplicate rate",
    })
    fig = px.bar(
        melted,
        x="dataset",
        y="value",
        color="risk",
        barmode="group",
        color_discrete_sequence=[WARN, FAIL],
        text_auto=".2%",
        title="Query repetition risk",
        labels={"dataset": "Corpus", "value": "Share", "risk": "Risk metric"},
    )
    fig.update_yaxes(tickformat=".0%", gridcolor=GRID)
    fig.update_xaxes(title="")
    return _layout(fig, legend_horizontal=True)


def build_entity_diversity_chart(df: pd.DataFrame) -> go.Figure:
    melted = df.melt(
        id_vars=["dataset", "rows", "unique_names"],
        value_vars=["name_unique_ratio", "name_skeleton_ratio"],
        var_name="metric",
        value_name="value",
    )
    melted["metric"] = melted["metric"].map({
        "name_unique_ratio": "Unique-name ratio",
        "name_skeleton_ratio": "Name-skeleton ratio",
    })
    fig = px.bar(
        melted,
        x="dataset",
        y="value",
        color="metric",
        barmode="group",
        color_discrete_sequence=[PALETTE[0], PALETTE[2]],
        text_auto=".1%",
        title="Entity naming diversity",
        labels={"dataset": "Inventory", "value": "Diversity ratio", "metric": "Metric"},
        hover_data={"rows": ":,", "unique_names": ":,"},
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05], gridcolor=GRID)
    fig.update_xaxes(title="")
    return _layout(fig, legend_horizontal=True)


def build_entity_risk_chart(df: pd.DataFrame) -> go.Figure:
    melted = df.melt(
        id_vars="dataset",
        value_vars=["largest_skeleton_share", "duplicate_name_rate"],
        var_name="risk",
        value_name="value",
    )
    melted["risk"] = melted["risk"].map({
        "largest_skeleton_share": "Largest skeleton share",
        "duplicate_name_rate": "Duplicate-name rate",
    })
    fig = px.bar(
        melted,
        x="dataset",
        y="value",
        color="risk",
        barmode="group",
        color_discrete_sequence=[WARN, FAIL],
        text_auto=".2%",
        title="Entity repetition risk",
        labels={"dataset": "Inventory", "value": "Share", "risk": "Risk metric"},
    )
    fig.update_yaxes(tickformat=".0%", gridcolor=GRID)
    fig.update_xaxes(title="")
    return _layout(fig, legend_horizontal=True)


def build_leakage_heatmap(leakage: dict[str, Any]) -> go.Figure:
    splits = sorted({part for key in leakage for part in key.split("__")})
    matrix = pd.DataFrame(0.0, index=splits, columns=splits)
    for s in splits:
        matrix.loc[s, s] = 0.0
    for key, metrics in leakage.items():
        a, b = key.split("__", 1)
        value = float(metrics.get("overlap_over_smaller_split", 0))
        matrix.loc[a, b] = value
        matrix.loc[b, a] = value
    fig = go.Figure(go.Heatmap(
        z=matrix.values,
        x=[x.title() for x in matrix.columns],
        y=[x.title() for x in matrix.index],
        zmin=0,
        zmax=max(0.02, float(matrix.values.max()) if matrix.size else 0.02),
        colorscale=[[0, "#DCFCE7"], [0.5, "#FEF3C7"], [1, "#FCA5A5"]],
        text=[[f"{v:.2%}" for v in row] for row in matrix.values],
        texttemplate="%{text}",
        hovertemplate="%{y} ↔ %{x}<br>Normalized overlap: %{z:.3%}<extra></extra>",
        colorbar=dict(title="Overlap"),
    ))
    fig.update_layout(title="Normalized pattern leakage between splits")
    return _layout(fig, height=430)


def build_router_label_chart(label_counts: dict[str, int]) -> go.Figure:
    if not label_counts:
        return _layout(go.Figure(), height=400)
    df = pd.DataFrame([{"intent": k, "count": v} for k, v in label_counts.items()]).sort_values("count", ascending=True)
    fig = px.bar(
        df,
        x="count",
        y="intent",
        orientation="h",
        color="count",
        color_continuous_scale="Viridis",
        text="count",
        title="Router intent-label balance",
        labels={"count": "Label occurrences", "intent": "Intent"},
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
    fig.update_coloraxes(showscale=False)
    fig.update_yaxes(title="")
    fig.update_xaxes(gridcolor=GRID)
    return _layout(fig, height=max(430, 46 * len(df) + 110))


def build_file_size_chart(files: dict[str, Any]) -> go.Figure:
    rows = [{"file": _human(name), "mb": float(meta.get("bytes", 0)) / (1024 * 1024)} for name, meta in files.items()]
    df = pd.DataFrame(rows).sort_values("mb", ascending=True)
    fig = px.bar(
        df,
        x="mb",
        y="file",
        orientation="h",
        color="mb",
        color_continuous_scale="Blues",
        text="mb",
        title="Audited file footprint",
        labels={"mb": "Size (MB)", "file": "File"},
    )
    fig.update_traces(texttemplate="%{text:.1f} MB", textposition="outside", cliponaxis=False)
    fig.update_coloraxes(showscale=False)
    fig.update_yaxes(title="")
    fig.update_xaxes(gridcolor=GRID)
    return _layout(fig, height=max(430, 38 * len(df) + 120))


def build_top_patterns_chart(report: dict[str, Any], dataset_key: str, top_n: int = 10) -> go.Figure:
    metrics = report.get("query_datasets", {}).get(dataset_key, {})
    rows = list(metrics.get("top_patterns", []))[:top_n]
    df = pd.DataFrame(rows)
    if df.empty:
        return _layout(go.Figure(), height=400)
    df = df.sort_values("count", ascending=True)
    fig = px.bar(
        df,
        x="count",
        y="pattern",
        orientation="h",
        color="count",
        color_continuous_scale="Sunsetdark",
        text="count",
        title=f"Most frequent normalized patterns · {_human(dataset_key)}",
        labels={"count": "Occurrences", "pattern": "Normalized pattern"},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_coloraxes(showscale=False)
    fig.update_yaxes(title="")
    fig.update_xaxes(gridcolor=GRID)
    return _layout(fig, height=max(430, 48 * len(df) + 120))


def render_data_quality_dashboard(st, root: Path, report: dict[str, Any], language: str, rerun_callback=None, chart_theme: str = "light") -> None:
    hu = language == "hu"
    def plot(fig, *, key: str, config: dict | None = None, width: str = "stretch") -> None:
        render_plotly(st, fig, key=key, theme=chart_theme, config=config)

    st.subheader("🧪 Adatminőség és leakage diagnosztika" if hu else "🧪 Data quality & leakage diagnostics")
    st.caption(
        "Interaktív minőségellenőrzés a nyelvi korpuszokra, entity inventorykra, train/validation/test leakage-re és quality gate-ekre."
        if hu else
        "Interactive quality diagnostics for language corpora, entity inventories, train/validation/test leakage, and explicit quality gates."
    )

    if rerun_callback is not None:
        if st.button("🔄 Teljes audit újrafuttatása" if hu else "🔄 Re-run full audit", key="dq_rerun_audit"):
            rerun_callback()
            st.rerun()

    gates = report.get("quality_gates", {})
    qdf = query_metrics_frame(report)
    edf = entity_metrics_frame(report)
    pass_rate = float(report.get("quality_gate_pass_rate", 0))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Quality gates", f"{sum(bool(v) for v in gates.values())}/{len(gates)}")
    k2.metric("Pass rate", f"{pass_rate:.0%}")
    k3.metric("Audited query rows", f"{int(qdf['rows'].sum()) if not qdf.empty else 0:,}")
    k4.metric("Audited entity rows", f"{int(edf['rows'].sum()) if not edf.empty else 0:,}")

    overview_tab, queries_tab, entities_tab, leakage_tab, gallery_tab, raw_tab = st.tabs([
        "🎯 Overview", "📝 Query corpora", "🏨 Entity inventories", "🧬 Leakage & balance", "🖼 Saved plots", "📄 Audit report"
    ])

    with overview_tab:
        left, right = st.columns([1, 1.5])
        with left:
            plot(build_quality_gauge(pass_rate), width="stretch", key="dq_overview_gauge", config={"displaylogo": False})
        with right:
            plot(build_gate_status_chart(gates), width="stretch", key="dq_overview_gates", config={"displaylogo": False})
        if report.get("files"):
            plot(build_file_size_chart(report["files"]), width="stretch", key="dq_overview_files", config={"displaylogo": False})

    with queries_tab:
        if not qdf.empty:
            left, right = st.columns(2)
            with left:
                plot(build_query_diversity_scatter(qdf), width="stretch", key="dq_query_diversity", config={"displaylogo": False, "scrollZoom": True})
            with right:
                plot(build_query_risk_chart(qdf), width="stretch", key="dq_query_risk", config={"displaylogo": False})

            dataset_keys = list(report.get("query_datasets", {}))
            if dataset_keys:
                c1, c2 = st.columns([2, 1])
                selected = c1.selectbox(
                    "Korpusz" if hu else "Corpus",
                    dataset_keys,
                    format_func=_human,
                    key="dq_pattern_dataset",
                )
                top_n = c2.slider("Top N", 5, 20, 10, key="dq_pattern_topn")
                plot(build_top_patterns_chart(report, selected, top_n), width="stretch", key="dq_top_patterns", config={"displaylogo": False})
            st.dataframe(qdf, width="stretch", hide_index=True)

    with entities_tab:
        if not edf.empty:
            left, right = st.columns(2)
            with left:
                plot(build_entity_diversity_chart(edf), width="stretch", key="dq_entity_diversity", config={"displaylogo": False})
            with right:
                plot(build_entity_risk_chart(edf), width="stretch", key="dq_entity_risk", config={"displaylogo": False})
            st.dataframe(edf, width="stretch", hide_index=True)

    with leakage_tab:
        leakage = report.get("split_leakage", {})
        if leakage:
            plot(build_leakage_heatmap(leakage), width="stretch", key="dq_leakage_heatmap", config={"displaylogo": False})
        label_counts = report.get("router_label_counts", {})
        if label_counts:
            plot(build_router_label_chart(label_counts), width="stretch", key="dq_router_label_balance", config={"displaylogo": False})
        if leakage:
            leak_rows = []
            for pair, metrics in leakage.items():
                leak_rows.append({"split_pair": pair.replace("__", " ↔ "), **metrics})
            st.dataframe(pd.DataFrame(leak_rows), width="stretch", hide_index=True)

    with gallery_tab:
        st.caption(
            "Az audit által mentett statikus PNG snapshotok. Az interaktív nézetek ugyanebből az audit reportból épülnek."
            if hu else
            "Static PNG snapshots saved by the audit. The interactive views are built from the same audit report."
        )
        gallery = [
            (root / "06_results/data_quality/query_pattern_diversity.png", "Query pattern diversity"),
            (root / "06_results/data_quality/entity_name_diversity.png", "Entity name diversity"),
        ]
        existing = [(p, title) for p, title in gallery if p.exists()]
        if not existing:
            st.info("Nincs mentett audit plot." if hu else "No saved audit plots found.")
        else:
            cols = st.columns(2)
            for col, (path, title) in zip(cols, existing):
                with col:
                    st.markdown(f"**{title}**")
                    st.image(str(path), width="stretch")

    with raw_tab:
        summary_path = root / "06_results/data_quality/data_quality_summary.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            st.dataframe(df, width="stretch", hide_index=True)
            st.download_button(
                "⬇ data_quality_summary.csv",
                data=df.to_csv(index=False).encode("utf-8-sig"),
                file_name="data_quality_summary.csv",
                mime="text/csv",
                key="dq_download_summary_csv",
            )
        st.download_button(
            "⬇ data_quality_report.json",
            data=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="data_quality_report.json",
            mime="application/json",
            key="dq_download_report_json",
        )

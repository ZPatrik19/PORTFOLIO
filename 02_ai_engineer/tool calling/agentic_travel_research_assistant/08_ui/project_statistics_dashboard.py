"""Interactive Plotly dashboard for the Project Statistics UI.

This module intentionally keeps chart construction separate from ``app.py`` so that
figures can be unit-tested without starting Streamlit. The dashboard reads only
reproducible repository artifacts under ``06_results/project_statistics`` plus a
small number of raw numeric columns for interactive histograms.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from chart_theme import render_plotly

# A restrained but clearly differentiated dashboard palette.
PALETTE = [
    "#2563EB",  # blue
    "#14B8A6",  # teal
    "#8B5CF6",  # violet
    "#F59E0B",  # amber
    "#EF4444",  # red
    "#06B6D4",  # cyan
    "#84CC16",  # lime
    "#EC4899",  # pink
]
QUALITY_COLORS = {"precision": "#2563EB", "recall": "#14B8A6", "f1": "#8B5CF6"}
METHOD_COLORS = {"rule_based": "#94A3B8", "plan_execute": "#F59E0B", "ml_router": "#2563EB"}


def _human_dataset(name: str) -> str:
    return name.replace(".csv", "").replace("_", " ").title()


def _human_method(name: str) -> str:
    return {
        "rule_based": "Rule-based baseline",
        "plan_execute": "Plan → Execute",
        "ml_router": "ML Router",
    }.get(name, name.replace("_", " ").title())


def _base_layout(fig: go.Figure, *, height: int = 460, legend_horizontal: bool = False) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=18, r=18, t=72, b=24),
        hoverlabel=dict(font_size=13),
        font=dict(size=13),
        title=dict(x=0.01, xanchor="left", font=dict(size=20)),
        legend=dict(title_text=""),
    )
    if legend_horizontal:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def build_dataset_scale_chart(df: pd.DataFrame, metric: str = "rows", log_scale: bool = False) -> go.Figure:
    work = df.copy()
    work["dataset_label"] = work["dataset"].map(_human_dataset)
    work = work.sort_values(metric, ascending=True)
    labels = {"rows": "Rows", "memory_mb": "Memory (MB)"}
    fig = px.bar(
        work,
        x=metric,
        y="dataset_label",
        orientation="h",
        color=metric,
        color_continuous_scale="Blues",
        text=metric,
        hover_data={"columns": True, "missing_cells": True, "exact_duplicate_rows": True, "dataset_label": False},
        title="Dataset scale" if metric == "rows" else "Dataset memory footprint",
        labels={metric: labels.get(metric, metric), "dataset_label": "Dataset"},
    )
    fig.update_traces(texttemplate="%{text:,.3~s}" if metric == "memory_mb" else "%{text:,.3s}", textposition="outside", cliponaxis=False)
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(type="log" if log_scale else "linear", gridcolor="#E2E8F0")
    fig.update_yaxes(title="")
    return _base_layout(fig, height=max(460, 34 * len(work) + 120))


def build_query_diversity_chart(df: pd.DataFrame) -> go.Figure:
    work = df.copy()
    work["dataset_label"] = work["dataset"].map(_human_dataset)
    melted = work.melt(
        id_vars=["dataset_label"],
        value_vars=["normalized_pattern_ratio", "multi_intent_rate"],
        var_name="metric",
        value_name="value",
    )
    melted["metric"] = melted["metric"].map({
        "normalized_pattern_ratio": "Linguistic diversity",
        "multi_intent_rate": "Multi-intent share",
    })
    fig = px.bar(
        melted,
        x="dataset_label",
        y="value",
        color="metric",
        barmode="group",
        color_discrete_sequence=[PALETTE[0], PALETTE[1]],
        text_auto=".1%",
        title="Query-corpus diversity and complexity",
        labels={"dataset_label": "Corpus", "value": "Share", "metric": "Metric"},
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05], gridcolor="#E2E8F0")
    fig.update_xaxes(title="")
    return _base_layout(fig, legend_horizontal=True)


def build_entity_diversity_chart(df: pd.DataFrame) -> go.Figure:
    work = df.copy()
    work["dataset_label"] = work["dataset"].map(_human_dataset)
    fig = px.bar(
        work,
        x="dataset_label",
        y="unique_name_ratio",
        color="dataset_label",
        color_discrete_sequence=PALETTE,
        text_auto=".1%",
        title="Entity-name diversity",
        labels={"dataset_label": "Entity inventory", "unique_name_ratio": "Unique-name ratio"},
        hover_data={"rows": ":,", "unique_names": ":,", "duplicate_names": ":,", "cities": True},
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05], gridcolor="#E2E8F0")
    fig.update_xaxes(title="")
    fig.update_layout(showlegend=False)
    return _base_layout(fig)


def build_intent_distribution_chart(df: pd.DataFrame) -> go.Figure:
    work = df.sort_values("count", ascending=True).copy()
    fig = px.bar(
        work,
        x="count",
        y="intent",
        orientation="h",
        color="count",
        color_continuous_scale="Tealgrn",
        text="count",
        title="Intent-label balance in the training corpus",
        labels={"count": "Label occurrences", "intent": "Intent"},
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
    fig.update_coloraxes(showscale=False)
    fig.update_yaxes(title="")
    fig.update_xaxes(gridcolor="#E2E8F0")
    return _base_layout(fig, height=max(420, 45 * len(work) + 120))


def build_complexity_donut(df: pd.DataFrame, x: str, y: str, title: str, label_prefix: str) -> go.Figure:
    work = df.copy()
    work["label"] = work[x].map(lambda v: f"{v} {label_prefix}" if int(v) != 1 else f"1 {label_prefix.rstrip('s')}")
    fig = px.pie(
        work,
        names="label",
        values=y,
        hole=0.58,
        color_discrete_sequence=PALETTE,
        title=title,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label", hovertemplate="%{label}<br>Cases: %{value:,}<br>Share: %{percent}<extra></extra>")
    fig.add_annotation(text=f"{int(work[y].sum()):,}<br><span style='font-size:12px'>total</span>", x=0.5, y=0.5, showarrow=False, font=dict(size=22))
    return _base_layout(fig, height=430)


def build_benchmark_tool_chart(df: pd.DataFrame) -> go.Figure:
    work = df.sort_values("expected_calls", ascending=True).copy()
    fig = px.bar(
        work,
        x="expected_calls",
        y="tool",
        orientation="h",
        color="expected_calls",
        color_continuous_scale="Viridis",
        text="expected_calls",
        title="Expected tool usage across benchmark cases",
        labels={"expected_calls": "Expected calls", "tool": "Tool"},
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
    fig.update_coloraxes(showscale=False)
    fig.update_yaxes(title="")
    fig.update_xaxes(gridcolor="#E2E8F0")
    return _base_layout(fig, height=max(430, 44 * len(work) + 110))


def build_category_chart(df: pd.DataFrame, title: str, top_n: int = 15) -> go.Figure:
    work = df.nlargest(top_n, "count").sort_values("count", ascending=True).copy()
    fig = px.bar(
        work,
        x="count",
        y="category",
        orientation="h",
        color="count",
        color_continuous_scale="Turbo",
        text="count",
        title=title,
        labels={"count": "Records", "category": "Category"},
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
    fig.update_coloraxes(showscale=False)
    fig.update_yaxes(title="")
    fig.update_xaxes(gridcolor="#E2E8F0")
    return _base_layout(fig, height=max(430, 36 * len(work) + 120))


def build_price_histogram(values: pd.Series, title: str, x_label: str, bins: int = 40) -> go.Figure:
    data = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce").dropna()})
    fig = px.histogram(
        data,
        x="value",
        nbins=bins,
        marginal="box",
        color_discrete_sequence=[PALETTE[0]],
        title=title,
        labels={"value": x_label, "count": "Records"},
    )
    fig.update_traces(hovertemplate=f"{x_label}: %{{x}}<br>Records: %{{y:,}}<extra></extra>")
    fig.update_xaxes(gridcolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#E2E8F0")
    return _base_layout(fig, height=500)


def build_router_metrics_chart(df: pd.DataFrame) -> go.Figure:
    work = df.melt(id_vars=["label"], value_vars=["precision", "recall", "f1"], var_name="metric", value_name="score")
    work["metric_label"] = work["metric"].map({"precision": "Precision", "recall": "Recall", "f1": "F1"})
    fig = px.bar(
        work,
        x="label",
        y="score",
        color="metric_label",
        barmode="group",
        color_discrete_map={"Precision": QUALITY_COLORS["precision"], "Recall": QUALITY_COLORS["recall"], "F1": QUALITY_COLORS["f1"]},
        title="Router quality by intent",
        labels={"label": "Intent", "score": "Score", "metric_label": "Metric"},
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05], gridcolor="#E2E8F0")
    fig.update_xaxes(title="", tickangle=-20)
    return _base_layout(fig, height=500, legend_horizontal=True)


def build_methodology_radar(df: pd.DataFrame) -> go.Figure:
    metrics = ["tool_selection_accuracy", "tool_f1", "argument_accuracy", "task_success"]
    labels = ["Exact tool selection", "Tool F1", "Argument accuracy", "Task success"]
    fig = go.Figure()
    for _, row in df.iterrows():
        name = str(row["methodology"])
        values = [float(row.get(m, 0) or 0) for m in metrics]
        values += values[:1]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels + labels[:1],
            fill="toself",
            name=_human_method(name),
            line=dict(color=METHOD_COLORS.get(name)),
            opacity=0.72,
            hovertemplate="%{theta}: %{r:.1%}<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%", gridcolor="#CBD5E1")),
        title="Methodology quality profile",
    )
    return _base_layout(fig, height=520, legend_horizontal=True)


def build_quality_latency_scatter(df: pd.DataFrame) -> go.Figure:
    work = df.copy()
    work["method_label"] = work["methodology"].map(_human_method)
    fig = px.scatter(
        work,
        x="latency_ms",
        y="tool_f1",
        size="num_calls",
        color="methodology",
        color_discrete_map=METHOD_COLORS,
        text="method_label",
        hover_data={
            "tool_selection_accuracy": ":.1%",
            "argument_accuracy": ":.1%",
            "task_success": ":.1%",
            "p95_latency_ms": ":.1f",
            "cases": ":.0f",
            "methodology": False,
            "method_label": False,
        },
        title="Quality vs latency trade-off",
        labels={"latency_ms": "Mean latency (ms)", "tool_f1": "Tool F1", "num_calls": "Avg calls"},
    )
    fig.update_traces(textposition="top center")
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05], gridcolor="#E2E8F0")
    fig.update_xaxes(gridcolor="#E2E8F0")
    fig.update_layout(showlegend=False)
    return _base_layout(fig, height=480)


def build_methodology_metric_chart(df: pd.DataFrame, metric: str) -> go.Figure:
    labels = {
        "tool_selection_accuracy": "Exact tool-selection accuracy",
        "tool_f1": "Tool F1",
        "argument_accuracy": "Argument accuracy",
        "task_success": "Task success rate",
        "latency_ms": "Mean latency (ms)",
        "p95_latency_ms": "P95 latency (ms)",
        "unnecessary_tool_call_rate": "Unnecessary tool-call rate",
    }
    work = df.copy()
    work["method_label"] = work["methodology"].map(_human_method)
    work = work.sort_values(metric, ascending=False)
    fig = px.bar(
        work,
        x="method_label",
        y=metric,
        color="methodology",
        color_discrete_map=METHOD_COLORS,
        text=metric,
        title=labels.get(metric, metric.replace("_", " ").title()),
        labels={"method_label": "Methodology", metric: labels.get(metric, metric)},
    )
    is_rate = metric not in {"latency_ms", "p95_latency_ms"}
    fig.update_traces(texttemplate="%{text:.1%}" if is_rate else "%{text:.1f} ms", textposition="outside", cliponaxis=False)
    if is_rate:
        fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
    fig.update_layout(showlegend=False)
    return _base_layout(fig, height=440)


def load_stat_csv(stats_dir: Path, filename: str) -> pd.DataFrame:
    path = stats_dir / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_raw_numeric(root: Path, filename: str, column: str) -> pd.Series:
    path = root / "01_data" / "raw" / filename
    if not path.exists():
        return pd.Series(dtype=float)
    return pd.read_csv(path, usecols=[column])[column]


def render_dashboard(st: Any, root: Path, ps: dict, language: str, chart_theme: str = "light") -> None:
    """Render the interactive Project Statistics dashboard in Streamlit."""
    hu = language == "hu"
    def plot(fig, *, key: str, config: dict | None = None, width: str = "stretch") -> None:
        render_plotly(st, fig, key=key, theme=chart_theme, config=config)

    stats_dir = root / "06_results" / "project_statistics"
    repo = ps.get("repository", {})
    data_stats = ps.get("data", {})
    router_stats = ps.get("intent_router", {})
    quality_stats = ps.get("data_quality", {})

    datasets = load_stat_csv(stats_dir, "dataset_statistics.csv")
    queries = load_stat_csv(stats_dir, "query_statistics.csv")
    entities = load_stat_csv(stats_dir, "entity_statistics.csv")
    labels = load_stat_csv(stats_dir, "tool_label_distribution.csv")
    query_complexity = load_stat_csv(stats_dir, "query_complexity_distribution.csv")
    bench_complexity = load_stat_csv(stats_dir, "benchmark_complexity.csv")
    bench_tools = load_stat_csv(stats_dir, "benchmark_expected_tool_distribution.csv")
    per_label = load_stat_csv(stats_dir, "router_per_label_metrics.csv")
    evaluations = load_stat_csv(stats_dir, "evaluation_statistics.csv")

    overview_tab, data_tab, routing_tab, model_tab, gallery_tab, tables_tab = st.tabs([
        "🎯 Áttekintés" if hu else "🎯 Overview",
        "🗃 Adatok & diverzitás" if hu else "🗃 Data & diversity",
        "🧭 Routing & benchmark",
        "🧠 Modell & módszerek" if hu else "🧠 Model & methods",
        "🖼 Mentett plotok" if hu else "🖼 Saved plots",
        "📄 Adattáblák" if hu else "📄 Data tables",
    ])

    with overview_tab:
        st.markdown("### " + ("Projekt állapot egy képernyőn" if hu else "Project health at a glance"))
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Raw adatsorok" if hu else "Raw rows", f"{int(data_stats.get('total_raw_rows', 0)):,}")
        k2.metric("Toolok" if hu else "Tools", int(repo.get("registered_tools", 0)))
        k3.metric("Benchmark", f"{int(repo.get('benchmark_cases', 0)):,}")
        k4.metric("Router train", f"{int(router_stats.get('train_rows', 0)):,}")
        k5.metric("Test micro-F1", f"{float(router_stats.get('test_micro_f1') or 0):.3f}")
        k6.metric("Quality gates", f"{float(quality_stats.get('quality_gate_pass_rate') or 0):.0%}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Source files", int(repo.get("python_source_files", 0)))
        m2.metric("Source lines", f"{int(repo.get('python_source_lines', 0)):,}")
        m3.metric("Automated tests", int(repo.get("test_functions", 0)))
        m4.metric("Docs", int(repo.get("documentation_markdown_files", 0)))

        if not datasets.empty:
            chart_metric = st.radio(
                "Dataset nézet" if hu else "Dataset view",
                ["rows", "memory_mb"],
                format_func=lambda v: {"rows": "Rekordszám" if hu else "Rows", "memory_mb": "Memória (MB)" if hu else "Memory (MB)"}[v],
                horizontal=True,
                key="project_stats_dataset_metric",
            )
            log_scale = st.toggle("Log skála" if hu else "Log scale", value=False, key="project_stats_log_scale")
            plot(build_dataset_scale_chart(datasets, chart_metric, log_scale), width="stretch", key="ps_overview_dataset_scale", config={"displaylogo": False})

        if not evaluations.empty:
            left, right = st.columns(2)
            with left:
                plot(build_methodology_radar(evaluations), width="stretch", key="ps_overview_methodology_radar", config={"displaylogo": False})
            with right:
                plot(build_quality_latency_scatter(evaluations), width="stretch", key="ps_overview_quality_latency", config={"displaylogo": False})

    with data_tab:
        st.markdown("### " + ("Adatminőség, diverzitás és eloszlások" if hu else "Data quality, diversity and distributions"))
        if not queries.empty or not entities.empty:
            left, right = st.columns(2)
            with left:
                if not queries.empty:
                    plot(build_query_diversity_chart(queries), width="stretch", key="ps_data_query_diversity", config={"displaylogo": False})
            with right:
                if not entities.empty:
                    plot(build_entity_diversity_chart(entities), width="stretch", key="ps_data_entity_diversity", config={"displaylogo": False})

        st.markdown("#### " + ("Interaktív ár-eloszlás" if hu else "Interactive price distribution"))
        price_options = {
            "Hotels": ("hotels.csv", "nightly_eur", "Hotel nightly-price distribution", "EUR / night"),
            "Restaurants": ("restaurants.csv", "avg_meal_eur", "Restaurant meal-price distribution", "EUR / person"),
            "Attractions": ("attractions.csv", "ticket_eur", "Attraction ticket-price distribution", "EUR / ticket"),
        }
        pcol1, pcol2 = st.columns([2, 1])
        selected_price = pcol1.selectbox("Adattípus" if hu else "Dataset", list(price_options), key="project_stats_price_dataset")
        bins = pcol2.slider("Histogram bin", 20, 100, 45, 5, key="project_stats_hist_bins")
        filename, column, title, xlabel = price_options[selected_price]
        values = load_raw_numeric(root, filename, column)
        if not values.empty:
            plot(build_price_histogram(values, title, xlabel, bins), width="stretch", key="ps_data_price_histogram", config={"displaylogo": False, "scrollZoom": True})

        st.markdown("#### " + ("Kategória-eloszlások" if hu else "Category distributions"))
        cat_map = {
            "Restaurant cuisines": ("restaurant_cuisine_distribution.csv", "Restaurant cuisine mix"),
            "Attraction categories": ("attraction_category_distribution.csv", "Attraction category mix"),
            "Weather conditions": ("weather_condition_distribution.csv", "Weather-condition mix"),
        }
        c1, c2 = st.columns([2, 1])
        selected_cat = c1.selectbox("Nézet" if hu else "View", list(cat_map), key="project_stats_category_view")
        top_n = c2.slider("Top N", 5, 25, 15, key="project_stats_category_topn")
        cat_file, cat_title = cat_map[selected_cat]
        cat_df = load_stat_csv(stats_dir, cat_file)
        if not cat_df.empty:
            plot(build_category_chart(cat_df, cat_title, top_n), width="stretch", key="ps_data_category_distribution", config={"displaylogo": False})

    with routing_tab:
        st.markdown("### " + ("Routing egyensúly és workflow komplexitás" if hu else "Routing balance and workflow complexity"))
        if not labels.empty:
            plot(build_intent_distribution_chart(labels), width="stretch", key="ps_routing_intent_distribution", config={"displaylogo": False})

        left, right = st.columns(2)
        with left:
            if not query_complexity.empty:
                plot(build_complexity_donut(query_complexity, "intent_count", "queries", "Training-query intent complexity", "intents"), width="stretch", key="ps_routing_query_complexity", config={"displaylogo": False})
        with right:
            if not bench_complexity.empty:
                plot(build_complexity_donut(bench_complexity, "tool_calls", "cases", "Benchmark workflow complexity", "tool calls"), width="stretch", key="ps_routing_benchmark_complexity", config={"displaylogo": False})

        if not bench_tools.empty:
            plot(build_benchmark_tool_chart(bench_tools), width="stretch", key="ps_routing_benchmark_tools", config={"displaylogo": False})

    with model_tab:
        st.markdown("### " + ("Modell-diagnosztika és módszertanok" if hu else "Model diagnostics and methodology comparison"))
        if not per_label.empty:
            plot(build_router_metrics_chart(per_label), width="stretch", key="ps_model_router_metrics", config={"displaylogo": False})
        if not evaluations.empty:
            metric_options = ["tool_selection_accuracy", "tool_f1", "argument_accuracy", "task_success", "unnecessary_tool_call_rate", "latency_ms", "p95_latency_ms"]
            chosen_metric = st.selectbox(
                "Összehasonlítási metrika" if hu else "Comparison metric",
                metric_options,
                format_func=lambda m: {
                    "tool_selection_accuracy": "Exact tool selection",
                    "tool_f1": "Tool F1",
                    "argument_accuracy": "Argument accuracy",
                    "task_success": "Task success",
                    "unnecessary_tool_call_rate": "Unnecessary tool calls",
                    "latency_ms": "Mean latency",
                    "p95_latency_ms": "P95 latency",
                }[m],
                key="project_stats_method_metric",
            )
            plot(build_methodology_metric_chart(evaluations, chosen_metric), width="stretch", key="ps_model_methodology_metric", config={"displaylogo": False})
            left, right = st.columns(2)
            with left:
                plot(build_methodology_radar(evaluations), width="stretch", key="ps_model_methodology_radar", config={"displaylogo": False})
            with right:
                plot(build_quality_latency_scatter(evaluations), width="stretch", key="ps_model_quality_latency", config={"displaylogo": False})

        with st.expander("Router runtime / thresholds"):
            st.json(router_stats)

    with gallery_tab:
        st.markdown("### " + ("Előre generált statikus plotok" if hu else "Pre-generated static plot gallery"))
        st.caption(
            "Ezek a repositoryval együtt mentett PNG-k reprodukálható snapshotok. Az interaktív Plotly nézetek fölöttük dinamikusan épülnek a CSV/JSON eredményekből."
            if hu else
            "These repository PNGs are reproducible snapshots. The interactive Plotly views above are built dynamically from the same CSV/JSON artifacts."
        )
        gallery_names = [
            ("dataset_row_counts.png", "Dataset scale"),
            ("dataset_memory.png", "Dataset memory"),
            ("query_diversity.png", "Query diversity"),
            ("entity_name_diversity.png", "Entity-name diversity"),
            ("intent_label_distribution.png", "Intent distribution"),
            ("query_complexity.png", "Query complexity"),
            ("benchmark_complexity.png", "Benchmark complexity"),
            ("benchmark_tool_distribution.png", "Benchmark tool distribution"),
            ("inventory_city_coverage.png", "Inventory city coverage"),
            ("hotel_price_distribution.png", "Hotel price distribution"),
            ("restaurant_price_distribution.png", "Restaurant price distribution"),
            ("attraction_ticket_distribution.png", "Attraction ticket distribution"),
            ("restaurant_cuisine_distribution.png", "Cuisine distribution"),
            ("attraction_category_distribution.png", "Attraction categories"),
            ("weather_condition_distribution.png", "Weather conditions"),
            ("router_per_label_metrics.png", "Router per-label metrics"),
            ("methodology_quality.png", "Methodology quality"),
            ("methodology_latency.png", "Methodology latency"),
        ]
        existing = [(stats_dir / name, label) for name, label in gallery_names if (stats_dir / name).exists()]
        if not existing:
            st.info("Nincs mentett plot." if hu else "No saved plots found.")
        else:
            for i in range(0, len(existing), 2):
                cols = st.columns(2)
                for col, item in zip(cols, existing[i:i+2]):
                    path, label = item
                    with col:
                        st.markdown(f"**{label}**")
                        st.image(str(path), width="stretch")

    with tables_tab:
        st.markdown("### " + ("Statisztikai adattáblák" if hu else "Statistics data tables"))
        table_files = [
            "dataset_statistics.csv", "query_statistics.csv", "entity_statistics.csv",
            "tool_label_distribution.csv", "query_complexity_distribution.csv",
            "benchmark_complexity.csv", "benchmark_expected_tool_distribution.csv",
            "router_per_label_metrics.csv", "evaluation_statistics.csv",
        ]
        for filename in table_files:
            df = load_stat_csv(stats_dir, filename)
            if df.empty:
                continue
            with st.expander(filename):
                st.dataframe(df, width="stretch", hide_index=True)
                st.download_button(
                    "⬇ CSV letöltése" if hu else "⬇ Download CSV",
                    data=df.to_csv(index=False).encode("utf-8-sig"),
                    file_name=filename,
                    mime="text/csv",
                    key=f"download_{filename}",
                )

        json_path = stats_dir / "project_statistics.json"
        if json_path.exists():
            st.download_button(
                "⬇ project_statistics.json",
                data=json_path.read_bytes(),
                file_name="project_statistics.json",
                mime="application/json",
                key="download_project_stats_json",
            )

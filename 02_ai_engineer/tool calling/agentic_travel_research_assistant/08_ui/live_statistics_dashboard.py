"""Interactive Plotly dashboard for persistent live usage analytics."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from chart_theme import render_plotly
from plotly.subplots import make_subplots

PALETTE = ["#2563EB", "#14B8A6", "#8B5CF6", "#F59E0B", "#EF4444", "#06B6D4", "#84CC16", "#EC4899"]
SUCCESS = "#16A34A"
FAIL = "#DC2626"
GRID = "#E2E8F0"


def _layout(fig: go.Figure, *, height: int = 440, legend_horizontal: bool = False) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=18, r=18, t=72, b=28),
        hoverlabel=dict(font_size=13),
        font=dict(size=13),
        title=dict(x=0.01, xanchor="left", font=dict(size=20)),
        legend=dict(title_text=""),
    )
    if legend_horizontal:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def build_daily_volume_chart(daily: pd.DataFrame) -> go.Figure:
    work = daily.copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=work["date"], y=work["questions"], name="Questions", marker_color=PALETTE[0], opacity=0.82))
    fig.add_trace(go.Scatter(x=work["date"], y=work["tool_calls"], name="Tool calls", mode="lines+markers", line=dict(color=PALETTE[1], width=3), marker=dict(size=7)))
    fig.update_layout(title="Daily usage volume", barmode="group")
    fig.update_xaxes(title="", gridcolor=GRID)
    fig.update_yaxes(title="Count", gridcolor=GRID)
    return _layout(fig, height=430, legend_horizontal=True)


def build_success_latency_chart(daily: pd.DataFrame) -> go.Figure:
    work = daily.copy()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=work["date"], y=work["success_rate"] * 100,
        name="Run success", mode="lines+markers",
        line=dict(color=SUCCESS, width=3), marker=dict(size=7),
        hovertemplate="%{x}<br>Success: %{y:.1f}%<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=work["date"], y=work["avg_latency_ms"],
        name="Average latency", mode="lines+markers",
        line=dict(color=PALETTE[3], width=3, dash="dot"), marker=dict(size=7),
        hovertemplate="%{x}<br>Latency: %{y:.1f} ms<extra></extra>",
    ), secondary_y=True)
    fig.update_layout(title="Reliability and latency trend")
    fig.update_yaxes(title_text="Success rate", ticksuffix="%", range=[0, 105], secondary_y=False, gridcolor=GRID)
    fig.update_yaxes(title_text="Latency (ms)", secondary_y=True, showgrid=False)
    fig.update_xaxes(title="")
    return _layout(fig, height=430, legend_horizontal=True)


def build_tool_usage_chart(tools: pd.DataFrame) -> go.Figure:
    work = tools.copy().sort_values("calls", ascending=True)
    work["success_rate"] = work.apply(lambda r: (float(r["successful_calls"]) / float(r["calls"])) if r["calls"] else 0, axis=1)
    fig = px.bar(
        work,
        x="calls", y="tool_name", orientation="h",
        color="success_rate", color_continuous_scale=[[0, "#F87171"], [0.75, "#FACC15"], [1, "#22C55E"]],
        range_color=[0, 1], text="calls",
        title="Tool usage and reliability",
        labels={"calls": "Calls", "tool_name": "Tool", "success_rate": "Success rate"},
        hover_data={"successful_calls": True, "failed_calls": True, "avg_latency_ms": ":.1f", "success_rate": ":.1%"},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_coloraxes(colorbar_tickformat=".0%")
    fig.update_yaxes(title="")
    fig.update_xaxes(gridcolor=GRID)
    return _layout(fig, height=max(430, 45 * len(work) + 120))


def build_tool_latency_box(history: pd.DataFrame) -> go.Figure:
    work = history.copy()
    fig = px.box(
        work,
        x="tool_name", y="latency_ms", color="tool_name",
        color_discrete_sequence=PALETTE,
        points="outliers",
        title="Tool latency distribution",
        labels={"tool_name": "Tool", "latency_ms": "Latency (ms)"},
        hover_data=["success", "methodology"],
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="")
    fig.update_yaxes(gridcolor=GRID)
    return _layout(fig, height=480)


def build_methodology_chart(methods: pd.DataFrame) -> go.Figure:
    work = methods.copy()
    work["success_pct"] = work["success_rate"] * 100
    fig = px.scatter(
        work,
        x="avg_latency_ms", y="success_pct",
        size="questions", color="methodology",
        color_discrete_sequence=PALETTE,
        text="methodology", size_max=46,
        title="Methodology quality vs latency",
        labels={"avg_latency_ms": "Average latency (ms)", "success_pct": "Run success rate"},
        hover_data={"questions": ":,", "avg_tool_calls": ":.2f", "multi_tool_rate": ":.1%"},
    )
    fig.update_traces(textposition="top center")
    fig.update_yaxes(ticksuffix="%", range=[0, 105], gridcolor=GRID)
    fig.update_xaxes(gridcolor=GRID)
    fig.update_layout(showlegend=False)
    return _layout(fig, height=450)


def build_destination_chart(destinations: pd.DataFrame) -> go.Figure:
    work = destinations.sort_values("tool_calls", ascending=False).copy()
    fig = px.treemap(
        work,
        path=["city"], values="tool_calls", color="tool_calls",
        color_continuous_scale="Tealgrn",
        title="Most-used destinations",
        hover_data={"tool_calls": ":,"},
    )
    fig.update_coloraxes(showscale=False)
    return _layout(fig, height=470)


def build_dimension_donut(df: pd.DataFrame, title: str) -> go.Figure:
    work = df.copy()
    fig = px.pie(
        work,
        names="value", values="questions", hole=0.58,
        color_discrete_sequence=PALETTE,
        title=title,
    )
    fig.update_traces(textinfo="percent+label", hovertemplate="%{label}<br>Questions: %{value:,}<br>Share: %{percent}<extra></extra>")
    total = int(work["questions"].sum()) if not work.empty else 0
    fig.add_annotation(text=f"{total:,}<br><span style='font-size:12px'>runs</span>", x=0.5, y=0.5, showarrow=False, font=dict(size=20))
    return _layout(fig, height=400)


def build_sequence_chart(sequences: pd.DataFrame) -> go.Figure:
    work = sequences.sort_values("runs", ascending=True).copy()
    fig = px.bar(
        work,
        x="runs", y="tool_sequence", orientation="h",
        color="runs", color_continuous_scale="Purples", text="runs",
        title="Most common tool sequences",
        labels={"runs": "Runs", "tool_sequence": "Tool sequence"},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_coloraxes(showscale=False)
    fig.update_yaxes(title="")
    fig.update_xaxes(gridcolor=GRID)
    return _layout(fig, height=max(430, 38 * len(work) + 130))


def build_activity_heatmap(interactions: pd.DataFrame) -> go.Figure:
    work = interactions.copy()
    dt = pd.to_datetime(work["created_at_utc"], utc=True, errors="coerce")
    work = work.assign(weekday=dt.dt.day_name(), hour=dt.dt.hour).dropna(subset=["weekday", "hour"])
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = work.groupby(["weekday", "hour"]).size().unstack(fill_value=0).reindex(weekdays, fill_value=0)
    pivot = pivot.reindex(columns=range(24), fill_value=0)
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=pivot.index,
        colorscale="Blues",
        hovertemplate="%{y} %{x}<br>Questions: %{z}<extra></extra>",
        colorbar=dict(title="Questions"),
    ))
    fig.update_layout(title="Usage heatmap by weekday and UTC hour")
    fig.update_xaxes(dtick=2)
    return _layout(fig, height=430)


def build_question_latency_scatter(interactions: pd.DataFrame) -> go.Figure:
    work = interactions.copy()
    work["question_length"] = work["question"].astype(str).str.len()
    work["status"] = work["run_success"].map({1: "Success", 0: "Failed"}).fillna("Unknown")
    fig = px.scatter(
        work,
        x="question_length", y="total_latency_ms",
        color="methodology", symbol="status",
        color_discrete_sequence=PALETTE,
        opacity=0.75,
        title="Question length vs end-to-end latency",
        labels={"question_length": "Question length (characters)", "total_latency_ms": "Latency (ms)"},
        hover_data=["source", "language", "data_mode", "tool_call_count"],
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return _layout(fig, height=460)


def render_live_statistics_dashboard(st, usage_store, language: str, chart_theme: str = "light") -> None:
    hu = language == "hu"
    def plot(fig, *, key: str, config: dict | None = None, width: str = "stretch") -> None:
        render_plotly(st, fig, key=key, theme=chart_theme, config=config)

    st.subheader("📈 Élő használati analitika" if hu else "📈 Live usage analytics")
    st.caption(
        "A dashboard a helyi SQLite historyból számolódik és minden Chat futás után változik. Az adatok nem kerülnek külső analytics szolgáltatáshoz."
        if hu else
        "The dashboard is computed from the local SQLite history and changes after every Chat run. Usage history is not sent to an external analytics service."
    )

    stats = usage_store.summary()
    total = int(stats.get("total_questions") or 0)
    if total == 0:
        st.info(
            "Még nincs eltárolt futás. Futtass egy saját vagy preset kérdést a Chat fülön, majd térj vissza ide."
            if hu else
            "No stored runs yet. Run a custom or preset question in Chat, then return here."
        )
        return

    period = usage_store.period_comparison(7)
    current, previous = period["current"], period["previous"]
    def delta(curr: float, prev: float, pct: bool = False) -> str | None:
        if prev == 0:
            return None
        change = (curr - prev) / abs(prev)
        return f"{change:+.1%}" if pct else f"{curr - prev:+,.0f}"

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Kérdések" if hu else "Questions", f"{total:,}", delta(int(current.get("q", 0)), int(previous.get("q", 0))))
    k2.metric("Tool hívások" if hu else "Tool calls", f"{int(stats['total_tool_calls']):,}", delta(int(current.get("tc", 0)), int(previous.get("tc", 0))))
    k3.metric("Tool sikerarány" if hu else "Tool success", f"{float(stats['tool_success_rate']):.1%}")
    k4.metric("Futás sikerarány" if hu else "Run success", f"{float(stats['run_success_rate']):.1%}", delta(float(current.get("sr", 0)), float(previous.get("sr", 0)), True))
    k5.metric("P95 latency", f"{float(stats['run_latency_p95_ms']):.1f} ms")

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Átlag tool/kérdés" if hu else "Avg tools/question", f"{float(stats['avg_tool_calls']):.2f}")
    s2.metric("Multi-tool arány" if hu else "Multi-tool rate", f"{float(stats['multi_tool_rate']):.1%}")
    s3.metric("Tool nélküli arány" if hu else "No-tool rate", f"{float(stats['no_tool_rate']):.1%}")
    s4.metric("Egyedi kérdések" if hu else "Unique questions", f"{float(stats['unique_question_rate']):.1%}")
    s5.metric("P50 latency", f"{float(stats['run_latency_p50_ms']):.1f} ms")

    overview_tab, trends_tab, tools_tab, behavior_tab, history_tab = st.tabs([
        "🎯 Overview", "📅 Trends", "🛠 Tools", "🧭 Behavior", "📜 History & export"
    ])

    daily = usage_store.daily_usage(30)
    tools = usage_store.tool_usage()
    methods = usage_store.methodology_usage()
    interactions = usage_store.export_interactions()
    tool_history = usage_store.tool_call_history()
    sequences = usage_store.tool_sequences(15)
    destinations = usage_store.destination_usage(15)
    errors = usage_store.error_breakdown(20)

    with overview_tab:
        if not daily.empty:
            left, right = st.columns(2)
            with left:
                plot(build_daily_volume_chart(daily), width="stretch", key="live_overview_daily_volume", config={"displaylogo": False})
            with right:
                plot(build_success_latency_chart(daily), width="stretch", key="live_overview_success_latency", config={"displaylogo": False})
        if not tools.empty:
            plot(build_tool_usage_chart(tools), width="stretch", key="live_overview_tool_usage", config={"displaylogo": False})

    with trends_tab:
        days = st.slider("Időablak (nap)" if hu else "Time window (days)", 7, 90, 30, key="live_trend_days")
        trend = usage_store.daily_usage(days)
        if not trend.empty:
            plot(build_daily_volume_chart(trend), width="stretch", key="live_trends_volume", config={"displaylogo": False, "scrollZoom": True})
            plot(build_success_latency_chart(trend), width="stretch", key="live_trends_quality_latency", config={"displaylogo": False})
        if not interactions.empty:
            plot(build_activity_heatmap(interactions), width="stretch", key="live_trends_activity_heatmap", config={"displaylogo": False})

    with tools_tab:
        if not tools.empty:
            plot(build_tool_usage_chart(tools), width="stretch", key="live_tools_usage", config={"displaylogo": False})
        if not tool_history.empty:
            plot(build_tool_latency_box(tool_history), width="stretch", key="live_tools_latency_box", config={"displaylogo": False})
        if not sequences.empty:
            plot(build_sequence_chart(sequences), width="stretch", key="live_tools_sequences", config={"displaylogo": False})
        if errors.empty:
            st.success("Nincs eltárolt tool hiba." if hu else "No stored tool errors.")
        else:
            st.markdown("#### Tool hibák" if hu else "#### Tool errors")
            st.dataframe(errors, width="stretch", hide_index=True)

    with behavior_tab:
        if not methods.empty:
            plot(build_methodology_chart(methods), width="stretch", key="live_behavior_methodology", config={"displaylogo": False})
        if not destinations.empty:
            plot(build_destination_chart(destinations), width="stretch", key="live_behavior_destinations", config={"displaylogo": False})
        if not interactions.empty:
            plot(build_question_latency_scatter(interactions), width="stretch", key="live_behavior_question_latency", config={"displaylogo": False})

        ldf = usage_store.dimension_usage("language")
        ddf = usage_store.dimension_usage("data_mode")
        sdf = usage_store.dimension_usage("source")
        cols = st.columns(3)
        if not ldf.empty:
            with cols[0]: plot(build_dimension_donut(ldf, "Language mix"), width="stretch", key="live_behavior_language", config={"displaylogo": False})
        if not ddf.empty:
            with cols[1]: plot(build_dimension_donut(ddf, "Data-mode mix"), width="stretch", key="live_behavior_data_mode", config={"displaylogo": False})
        if not sdf.empty:
            with cols[2]: plot(build_dimension_donut(sdf, "Question-source mix"), width="stretch", key="live_behavior_source", config={"displaylogo": False})

    with history_tab:
        recent_limit = st.slider("Megjelenített futások" if hu else "Runs to display", 10, 250, 50, 10, key="live_history_limit")
        recent = usage_store.recent_interactions(recent_limit)
        st.dataframe(recent, width="stretch", hide_index=True)
        export_df = usage_store.export_interactions()
        st.download_button(
            "⬇ Usage history CSV" if not hu else "⬇ Használati előzmények CSV",
            data=export_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="travel_agent_usage_history.csv",
            mime="text/csv",
            key="live_download_history",
        )
        with st.expander("🗑 History törlése" if hu else "🗑 Clear history"):
            st.warning(
                "Ez csak a helyi usage_history.sqlite3 adatbázist törli; a modelleket és travel dataseteket nem."
                if hu else
                "This only clears the local usage_history.sqlite3 database; models and travel datasets are not affected."
            )
            confirm = st.checkbox("Igen, törölhető" if hu else "Yes, clear history", key="live_confirm_clear")
            if st.button("History törlése" if hu else "Clear history", disabled=not confirm, key="live_clear_button"):
                usage_store.clear()
                st.success("History törölve." if hu else "History cleared.")
                st.rerun()

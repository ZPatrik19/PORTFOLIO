from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix

from prompt_benchmark.paths import PATHS
from prompt_benchmark.benchmark.runner import run_strategy
from prompt_benchmark.config import load_yaml
from prompt_benchmark.constants import LABELS
from prompt_benchmark.evaluation.metrics import classification_metrics, confusion_table, per_class_report
from prompt_benchmark.llm.factory import PROVIDER_CAPABILITIES, SUPPORTED_PROVIDERS, create_llm_client
from prompt_benchmark.prompts import (TECHNIQUE_CATALOG, CustomPromptStrategy, get_strategy, list_custom_prompts, list_strategies, load_custom_prompt, save_custom_prompt)
from prompt_benchmark.utils.pricing import get_provider_pricing
from prompt_benchmark.ui.benchmark_panel import render_benchmark_panel, _performance_figure, _efficiency_figure, _difficulty_figure
from prompt_benchmark.ui.dataset_panel import render_dataset_panel
from prompt_benchmark.ui.playground_panel import render_playground_panel
from prompt_benchmark.ui.workflow_panel import render_workflow_banner, render_history_panel
from prompt_benchmark.ui.run_history import latest_run_id, load_summary as load_history_summary, load_raw as load_history_raw

os.chdir(PROJECT_ROOT)
load_dotenv()

cfg = load_yaml(PATHS.configs / "benchmark.yaml")
pricing = load_yaml(PATHS.configs / "pricing.yaml")
STRATEGIES = list_strategies()
LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, UnicodeDecodeError, ValueError, pd.errors.ParserError) as exc:
        LOGGER.warning("Could not read CSV %s: %s", path, exc)
        return None


def strategy_title(strategy: str) -> str:
    meta = TECHNIQUE_CATALOG.get(strategy, {})
    return f"{strategy.split('_')[0].upper()} · {meta.get('title', strategy)}"


def enrich_strategy_names(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "strategy" in frame.columns:
        frame["strategy_title"] = frame["strategy"].map(strategy_title)
    return frame


def _active_history_run(provider_name: str) -> str | None:
    selected = st.session_state.get("selected_history_run_id")
    if selected:
        try:
            from prompt_benchmark.ui.run_history import load_manifest
            manifest = load_manifest(str(selected))
            if manifest.get("provider") == provider_name:
                return str(selected)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            LOGGER.warning("Ignoring unavailable selected history run %s: %s", selected, exc)
    return latest_run_id(provider_name)


def load_provider_summary(provider_name: str) -> pd.DataFrame | None:
    run_id = _active_history_run(provider_name)
    frame = load_history_summary(run_id) if run_id else None
    if frame is None:
        frame = safe_read_csv(PATHS.results / provider_name / "benchmark_summary.csv")
    return enrich_strategy_names(frame) if frame is not None else None


def load_provider_raw(provider_name: str, strategy_name: str) -> pd.DataFrame | None:
    run_id = _active_history_run(provider_name)
    frame = load_history_raw(run_id, strategy_name) if run_id else None
    if frame is not None:
        return frame
    return safe_read_csv(PATHS.results / "raw" / provider_name / f"{strategy_name}.csv")


def load_case_summary(provider_name: str) -> pd.DataFrame | None:
    run_id = _active_history_run(provider_name)
    if run_id:
        frame = safe_read_csv(PATHS.results / "history" / run_id / "case_type_summary.csv")
        if frame is not None:
            return frame
    return safe_read_csv(PATHS.results / provider_name / "case_type_summary.csv")


def load_difficulty_summary(provider_name: str) -> pd.DataFrame | None:
    run_id = _active_history_run(provider_name)
    if run_id:
        frame = safe_read_csv(PATHS.results / "history" / run_id / "difficulty_summary.csv")
        if frame is not None:
            return frame
    return safe_read_csv(PATHS.results / provider_name / "difficulty_summary.csv")


def load_prompt_complexity(provider_name: str) -> pd.DataFrame | None:
    return safe_read_csv(PATHS.results / provider_name / "prompt_complexity.csv")


def load_per_class(provider_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    root = PATHS.reports / "per_class" / provider_name
    if not root.exists():
        return pd.DataFrame()
    for path in sorted(root.glob("p*.csv")):
        frame = pd.read_csv(path, index_col=0)
        strategy = path.stem
        for label, row in frame.iterrows():
            rows.append(
                {
                    "strategy": strategy,
                    "strategy_title": strategy_title(strategy),
                    "label": label,
                    "precision": row.get("precision", np.nan),
                    "recall": row.get("recall", np.nan),
                    "f1": row.get("f1-score", np.nan),
                    "support": row.get("support", np.nan),
                }
            )
    return pd.DataFrame(rows)


def run_command(command: list[str]) -> tuple[int, str]:
    proc = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, output.strip()


# -----------------------------------------------------------------------------
# Plotly helpers
# -----------------------------------------------------------------------------
def metric_bar(summary: pd.DataFrame, metric: str, title: str, y_title: str) -> go.Figure:
    ordered = summary.sort_values(metric, ascending=False).copy()
    fig = px.bar(
        ordered,
        x="strategy_title",
        y=metric,
        hover_data=[
            c for c in ["model", "accuracy", "macro_f1", "p95_latency_seconds", "mean_total_tokens", "cost_per_1000_requests_usd"]
            if c in ordered.columns
        ],
        title=title,
    )
    fig.update_layout(xaxis_title="Prompt strategy", yaxis_title=y_title, xaxis_tickangle=-45)
    return fig


def f1_ci_chart(summary: pd.DataFrame) -> go.Figure:
    ordered = summary.sort_values("macro_f1", ascending=True).copy()
    plus = (ordered["macro_f1_ci_high"] - ordered["macro_f1"]).clip(lower=0)
    minus = (ordered["macro_f1"] - ordered["macro_f1_ci_low"]).clip(lower=0)
    fig = go.Figure(
        go.Scatter(
            x=ordered["macro_f1"],
            y=ordered["strategy_title"],
            mode="markers",
            error_x=dict(type="data", array=plus, arrayminus=minus, visible=True),
            customdata=np.column_stack([
                ordered["accuracy"],
                ordered["p95_latency_seconds"],
                ordered["mean_total_tokens"],
            ]),
            hovertemplate=(
                "<b>%{y}</b><br>Macro F1=%{x:.4f}<br>Accuracy=%{customdata[0]:.4f}"
                "<br>P95 latency=%{customdata[1]:.3f}s<br>Mean tokens=%{customdata[2]:.1f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(title="Macro F1 with 95% bootstrap confidence interval", xaxis_title="Macro F1", yaxis_title="")
    return fig


def tradeoff_scatter(summary: pd.DataFrame, x: str, x_title: str, title: str) -> go.Figure:
    size = summary["mean_total_tokens"].fillna(1).clip(lower=1)
    fig = px.scatter(
        summary,
        x=x,
        y="macro_f1",
        size=size,
        hover_name="strategy_title",
        hover_data={
            "accuracy": ":.4f",
            "macro_f1": ":.4f",
            "p95_latency_seconds": ":.3f",
            "mean_total_tokens": ":.1f",
            "cost_per_1000_requests_usd": ":.6f",
            "model": True,
        },
        title=title,
    )
    fig.update_layout(xaxis_title=x_title, yaxis_title="Macro F1")
    return fig


def per_class_heatmap(provider_name: str) -> go.Figure | None:
    frame = load_per_class(provider_name)
    if frame.empty:
        return None
    pivot = frame.pivot(index="strategy_title", columns="label", values="f1")
    desired = [label for label in LABELS if label in pivot.columns]
    pivot = pivot.reindex(columns=desired)
    return px.imshow(
        pivot,
        text_auto=".3f",
        aspect="auto",
        zmin=0,
        zmax=1,
        labels=dict(x="Class", y="Prompt strategy", color="F1"),
        title="Per-class F1 heatmap",
    )


def case_type_heatmap(provider_name: str, metric: str = "accuracy") -> go.Figure | None:
    frame = load_case_summary(provider_name)
    if frame is None or frame.empty or metric not in frame.columns:
        return None
    frame = frame.copy()
    frame["strategy_title"] = frame["strategy"].map(strategy_title)
    pivot = frame.pivot(index="strategy_title", columns="case_type", values=metric)
    return px.imshow(
        pivot, text_auto=".3f", aspect="auto", zmin=0, zmax=1,
        labels=dict(x="Synthetic case type", y="Prompt strategy", color=metric.replace("_", " ").title()),
        title=f"Robustness by synthetic case type — {metric.replace('_', ' ').title()}",
    )


def difficulty_chart(provider_name: str) -> go.Figure | None:
    frame = load_difficulty_summary(provider_name)
    if frame is None or frame.empty:
        return None
    frame = frame.copy()
    frame["strategy_title"] = frame["strategy"].map(strategy_title)
    return px.bar(
        frame, x="strategy_title", y="accuracy", color="difficulty", barmode="group",
        hover_data=["macro_f1", "requests", "mean_total_tokens", "p95_latency_seconds"],
        title="Accuracy by difficulty level",
    )


def confusion_figure(frame: pd.DataFrame, title: str) -> go.Figure:
    # Invalid/unparseable responses are represented as a dedicated column instead
    # of NaN. sklearn.confusion_matrix rejects a mix of strings and NaN/unknown
    # targets, which previously crashed the Output Validation tab.
    matrix = confusion_table(frame)
    return px.imshow(
        matrix,
        x=list(matrix.columns),
        y=list(matrix.index),
        text_auto=True,
        aspect="auto",
        labels=dict(x="Predicted label", y="True label", color="Count"),
        title=title,
    )


def custom_options() -> dict[str, Path]:
    return {p.stem: p for p in list_custom_prompts()}


def build_custom_from_state(prefix: str = "custom") -> CustomPromptStrategy:
    return CustomPromptStrategy(
        display_name=st.session_state.get(f"{prefix}_name", "custom_prompt"),
        system_prompt=st.session_state.get(f"{prefix}_system", ""),
        user_template=st.session_state.get(
            f"{prefix}_user",
            "Task: classify the primary intent of this support ticket.\nAllowed labels: api, billing, cancellation, complaint, technical, upgrade.\n\n<ticket>\n{ticket}\n</ticket>\n\nReturn only the final label.",
        ),
        output_mode=st.session_state.get(f"{prefix}_output_mode", "label"),
        structured_output=bool(st.session_state.get(f"{prefix}_structured", False)),
        reasoning_effort=(st.session_state.get(f"{prefix}_reasoning") or None),
    )


def parameter_sweep_figure(sweep: pd.DataFrame, metric: str = "macro_f1") -> go.Figure:
    ordered = sweep.sort_values(["parameter", "value"])
    fig = px.line(
        ordered,
        x="value",
        y=metric,
        color="parameter",
        markers=True,
        hover_data=["model", "strategy", "mean_total_tokens", "p95_latency_seconds", "invalid_output_rate"],
        title=f"Decoding parameter sensitivity — {metric}",
    )
    fig.update_layout(xaxis_title="Parameter value", yaxis_title=metric.replace("_", " ").title())
    return fig


def token_latency_box(raw_frames: dict[str, pd.DataFrame], metric: str) -> go.Figure | None:
    rows: list[pd.DataFrame] = []
    for strategy, frame in raw_frames.items():
        if metric not in frame.columns:
            continue
        part = frame[[metric]].copy()
        part["strategy_title"] = strategy_title(strategy)
        rows.append(part)
    if not rows:
        return None
    data = pd.concat(rows, ignore_index=True)
    return px.box(data, x="strategy_title", y=metric, points=False, title=f"{metric.replace('_', ' ').title()} distribution")


# -----------------------------------------------------------------------------
# Header — runtime/provider settings are managed by the shared 11_ui_app.py.
# -----------------------------------------------------------------------------
st.title("🧪 Prompt Engineering Benchmark Lab")
st.caption(
    "Interactive Prompt Engineering, decoding-parameter, robustness and API-integration laboratory. "
    "Mock mode is a deterministic simulation; cloud/local providers produce real model measurements."
)

def client_overrides() -> dict[str, object]:
    values: dict[str, object] = {"model": model}
    if temperature_enabled:
        values["temperature"] = temperature
    if top_p_enabled:
        values["top_p"] = top_p
    if top_k_enabled:
        values["top_k"] = top_k
    return values


def build_client(extra_overrides: dict[str, object] | None = None):
    if provider == "openai" and (not model.strip() or model.strip().startswith("<")):
        raise RuntimeError(
            "OpenAI provider selected, but no model ID is configured. "
            "Set OPENAI_MODEL in .env or enter a valid current model ID in the sidebar."
        )
    overrides = client_overrides()
    overrides.update(extra_overrides or {})
    return create_llm_client(provider, cfg, overrides=overrides)


# -----------------------------------------------------------------------------
# Main tabs
# -----------------------------------------------------------------------------
_provider_ready = provider in {"mock", "ollama"} or bool(os.getenv(KEY_ENV.get(provider, "")))
render_workflow_banner("en", _provider_ready and bool(model))

(
    tab_dataset,
    tab_playground,
    tab_benchmark,
    tab_validation,
    tab_dashboard,
    tab_history,
    tab_techniques,
    tab_custom,
    tab_sweep,
    tab_finetune,
    tab_api,
    tab_system,
) = st.tabs(
    [
        "2️⃣ Dataset",
        "3️⃣ Playground",
        "4️⃣ Benchmark",
        "5️⃣ Output Validation",
        "6️⃣ Dashboard",
        "7️⃣ History",
        "🧠 Techniques",
        "✍️ Custom Prompt",
        "🎛 Parameter Lab",
        "🧬 Fine-tuning Prep",
        "🔌 API Integration",
        "🛠 System",
    ]
)

with tab_dataset:
    render_dataset_panel("en")


with tab_dashboard:
    st.subheader(f"Saved benchmark dashboard — {provider}")
    active_run = _active_history_run(provider)
    h1, h2 = st.columns([5, 1])
    h1.caption(f"Active history run: `{active_run or 'legacy/latest'}`")
    if h2.button("🔄 Refresh", key="en_dashboard_refresh_v3"):
        st.rerun()
    if provider == "mock":
        st.info(
            "The saved Mock dashboard is a deterministic simulation designed to behave like a prompt benchmark. "
            "Its quality differences are synthetic; token counts are estimated and latency is simulated. "
            "Switch to Ollama/Groq/Gemini/OpenAI to collect real model measurements with the same pipeline."
        )

    summary = load_provider_summary(provider)
    if summary is None or summary.empty:
        st.info(
            "No saved benchmark summary exists for this provider yet. Run the Benchmark Runner or use the CLI, "
            "then generate the report."
        )
    else:
        best = summary.sort_values("macro_f1", ascending=False).iloc[0]
        baseline_rows = summary[summary["strategy"] == "p0_zero_shot"]
        baseline = baseline_rows.iloc[0] if not baseline_rows.empty else best
        cheapest = summary.sort_values("cost_per_1000_requests_usd", ascending=True).iloc[0]
        fastest = summary.sort_values("p95_latency_seconds", ascending=True).iloc[0]
        most_reliable = summary.sort_values(["invalid_output_rate", "macro_f1"], ascending=[True, False]).iloc[0]
        best_delta_pp = (float(best["macro_f1"]) - float(baseline["macro_f1"])) * 100.0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Best Macro F1", f"{best['macro_f1']:.4f}", strategy_title(str(best["strategy"])))
        c2.metric("Baseline Macro F1", f"{baseline['macro_f1']:.4f}", "P0 Zero-shot")
        c3.metric("Best improvement", f"{best_delta_pp:+.2f} pp", "vs P0")
        c4.metric("Best output validity", f"{1 - most_reliable['invalid_output_rate']:.1%}", strategy_title(str(most_reliable["strategy"])))
        c5.metric("Fastest P95", f"{fastest['p95_latency_seconds']:.3f}s", strategy_title(str(fastest["strategy"])))

        total_requests = int(summary["requests"].sum()) if "requests" in summary else 0
        total_tokens_all = int(summary["total_tokens"].sum()) if "total_tokens" in summary else 0
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Best mean tokens", f"{float(best['mean_total_tokens']):.0f}", "per request")
        c2.metric("Best P95 latency", f"{float(best['p95_latency_seconds']):.3f}s")
        c3.metric("Best invalid rate", f"{float(best['invalid_output_rate']):.1%}")
        c4.metric("Saved requests", f"{total_requests:,}", "all strategies")
        c5.metric("Saved total tokens", f"{total_tokens_all:,}", "all strategies")

        with st.expander("Benchmark configuration and measurement sources", expanded=True):
            settings_cols = [
                "strategy", "model", "temperature", "top_p", "top_k", "token_source", "latency_source",
                "requests", "mean_branch_count",
            ]
            settings = summary[[c for c in settings_cols if c in summary.columns]].copy()
            if "strategy" in settings:
                settings.insert(1, "strategy_title", settings["strategy"].map(strategy_title))
            st.dataframe(settings, use_container_width=True, hide_index=True)
            st.caption(
                "For real providers, token counts come from the provider response when available and latency is wall-clock request time. "
                "For Mock, these fields are explicitly marked estimated_mock / simulated_mock."
            )

        sample_n = int(best.get("requests", 0))
        if {"accuracy_ci_low", "accuracy_ci_high"}.issubset(summary.columns):
            st.info(
                f"Active run: n={sample_n} / strategy · best Accuracy={best['accuracy']:.3f} · "
                f"Wilson 95% CI=[{best['accuracy_ci_low']:.3f}, {best['accuracy_ci_high']:.3f}]. "
                + ("This is still a pilot sample; a 1.000 score is not a stable final result." if sample_n < 60 else "The sample is more informative, but inspect difficulty/scenario slices as well.")
            )
        left, right = st.columns(2)
        with left:
            st.plotly_chart(_performance_figure(summary, "en"), use_container_width=True)
        with right:
            st.plotly_chart(_efficiency_figure(summary, "en"), use_container_width=True)
        diff_fig = _difficulty_figure(summary, "en")
        if diff_fig is not None:
            st.plotly_chart(diff_fig, use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                tradeoff_scatter(summary, "cost_per_1000_requests_usd", "Estimated USD / 1K requests", "Quality vs cost"),
                use_container_width=True,
            )
        with right:
            st.plotly_chart(
                tradeoff_scatter(summary, "p95_latency_seconds", "P95 latency (seconds)", "Quality vs latency"),
                use_container_width=True,
            )

        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                tradeoff_scatter(summary, "mean_total_tokens", "Mean total tokens / request", "Prompt/token overhead vs quality"),
                use_container_width=True,
            )
        with right:
            st.plotly_chart(metric_bar(summary, "invalid_output_rate", "Invalid output rate", "Invalid output rate"), use_container_width=True)

        st.markdown("### Robustness across realistic synthetic cases")
        case_heatmap = case_type_heatmap(provider, "accuracy")
        difficulty_fig = difficulty_chart(provider)
        if case_heatmap is not None:
            st.plotly_chart(case_heatmap, use_container_width=True)
        if difficulty_fig is not None:
            st.plotly_chart(difficulty_fig, use_container_width=True)
        if case_heatmap is None and provider != "mock":
            st.caption("Case-type breakdown is available when the dataset contains case_type metadata (the bundled synthetic dataset does).")

        paired_path = (PATHS.results / "history" / active_run / "paired_vs_p0.csv") if active_run else None
        paired = safe_read_csv(paired_path) if paired_path else None
        if paired is not None and not paired.empty:
            if "strategy" in paired:
                paired["strategy_title"] = paired["strategy"].map(strategy_title)
            st.markdown("### Paired comparison against P0 baseline")
            st.caption("On the exact same samples: how many baseline errors were fixed, and how many previously-correct samples regressed.")
            st.dataframe(paired, use_container_width=True, hide_index=True)
            st.plotly_chart(px.scatter(paired, x="regressed_vs_p0", y="fixed_vs_p0", color="paired_accuracy_delta_pp", hover_name="strategy_title", title="Fixed vs regressed samples"), use_container_width=True)

        heatmap = per_class_heatmap(provider)
        if heatmap is not None:
            st.plotly_chart(heatmap, use_container_width=True)

        st.markdown("#### Interactive leaderboard — key quality + efficiency numbers")
        complexity = load_prompt_complexity(provider)
        leaderboard = summary.copy()
        if complexity is not None and not complexity.empty:
            leaderboard = leaderboard.merge(
                complexity[[c for c in ["strategy", "category", "approx_prompt_tokens", "structured_output", "output_mode", "reasoning_effort", "branch_count"] if c in complexity.columns]],
                on="strategy",
                how="left",
            )
        shown_cols = [
            "strategy_title", "category", "requests", "accuracy", "accuracy_ci_low", "accuracy_ci_high", "balanced_accuracy", "matthews_corrcoef", "cohen_kappa", "macro_precision", "macro_recall", "macro_f1", "hard_macro_f1",
            "absolute_f1_improvement_vs_p0", "relative_f1_improvement_vs_p0_pct",
            "output_contract_valid_rate", "invalid_json_rate",
            "approx_prompt_tokens", "mean_input_tokens", "mean_output_tokens", "mean_total_tokens", "total_tokens",
            "tokens_per_correct_prediction", "p50_latency_seconds", "p95_latency_seconds", "p99_latency_seconds",
            "cost_per_1000_requests_usd", "benchmark_cost_usd",
            "temperature", "top_p", "top_k", "structured_output", "reasoning_effort", "branch_count",
        ]
        st.dataframe(leaderboard[[c for c in shown_cols if c in leaderboard.columns]], use_container_width=True, hide_index=True)

        with st.expander("Static report figures (GitHub export)"):
            fig_dir = PATHS.figures / provider
            figures = sorted(fig_dir.glob("*.png")) if fig_dir.exists() else []
            selected_static = st.selectbox("Static figure", [p.name for p in figures] if figures else ["No figures found"])
            if figures:
                st.image(str(fig_dir / selected_static), use_container_width=True)


with tab_techniques:
    st.subheader("Prompt technique library + complete prompt gallery")
    catalog = pd.DataFrame(
        [{"strategy": name, "display": strategy_title(name), **TECHNIQUE_CATALOG[name]} for name in STRATEGIES]
    )
    category = st.multiselect("Filter by category", sorted(catalog["category"].unique()), default=sorted(catalog["category"].unique()))
    st.dataframe(catalog[catalog["category"].isin(category)], use_container_width=True, hide_index=True)

    sample = st.text_area(
        "Ticket used to render every prompt template",
        "I was charged twice, but my main request is to cancel before the next renewal.",
        key="preview_ticket",
    )
    view_mode = st.radio("Template view", ["Inspect one strategy", "Show all P0–P16 templates"], horizontal=True)

    def show_payload(strategy_name: str, expanded: bool = False) -> None:
        meta = TECHNIQUE_CATALOG[strategy_name]
        strategy_obj = get_strategy(strategy_name)
        with st.expander(f"{strategy_title(strategy_name)} — {meta['category']}", expanded=expanded):
            st.write(f"**Hypothesis:** {meta['hypothesis']}")
            if hasattr(strategy_obj, "build_branches"):
                payloads = strategy_obj.build_branches(sample)
            else:
                payloads = [strategy_obj.build(sample)]
            total_chars = sum(len((p.instructions or "") + p.input_text) for p in payloads)
            total_words = sum(len(((p.instructions or "") + " " + p.input_text).split()) for p in payloads)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Branches", len(payloads))
            c2.metric("Prompt chars", f"{total_chars:,}")
            c3.metric("Prompt words", f"{total_words:,}")
            c4.metric("Approx tokens", f"{total_chars / 4:.0f}")
            c5.metric("Structured", "Yes" if any(p.structured_output for p in payloads) else "No")
            for i, payload in enumerate(payloads, 1):
                if len(payloads) > 1:
                    st.markdown(f"**Branch {i}**")
                st.code(f"SYSTEM:\n{payload.instructions or '(none)'}\n\nUSER:\n{payload.input_text}", language="text")
                st.json(
                    {
                        "output_mode": payload.output_mode,
                        "structured_output": payload.structured_output,
                        "reasoning_effort": payload.reasoning_effort,
                        "temperature": temperature,
                        "top_p": top_p if top_p_enabled else None,
                        "top_k": top_k if top_k_enabled else None,
                    }
                )

    if view_mode == "Inspect one strategy":
        selected = st.selectbox("Inspect strategy", STRATEGIES, format_func=strategy_title, key="inspect_strategy")
        show_payload(selected, expanded=True)
    else:
        st.caption("All prompt templates below are generated from the exact strategy code used by the benchmark — not copied documentation snippets.")
        for idx, strategy_name in enumerate(STRATEGIES):
            show_payload(strategy_name, expanded=idx == 0)


# -----------------------------------------------------------------------------
# Custom prompt editor
# -----------------------------------------------------------------------------
with tab_custom:
    st.subheader("Custom prompt editor, presets and benchmark")
    st.write("The `{ticket}` placeholder is required. Escape literal JSON braces as `{{` and `}}` because the template uses Python formatting.")

    st.session_state.setdefault("custom_name", "my_advanced_prompt")
    st.session_state.setdefault("custom_system", "You are a precise SaaS support-routing classifier.")
    st.session_state.setdefault(
        "custom_user",
        "Task: classify the primary intent of this support ticket.\nAllowed labels: api, billing, cancellation, complaint, technical, upgrade.\n\n<ticket>\n{ticket}\n</ticket>\n\nReturn only the final label.",
    )
    st.session_state.setdefault("custom_output_mode", "label")
    st.session_state.setdefault("custom_structured", False)
    st.session_state.setdefault("custom_reasoning", "")

    saved = custom_options()
    chosen_saved = st.selectbox("Load saved preset", ["(new prompt)", *saved.keys()], key="en_custom_saved_choice")
    if chosen_saved != "(new prompt)" and st.button("Load preset", key="en_custom_load_button"):
        loaded = load_custom_prompt(saved[chosen_saved])
        st.session_state["custom_name"] = loaded.display_name
        st.session_state["custom_system"] = loaded.system_prompt
        st.session_state["custom_user"] = loaded.user_template
        st.session_state["custom_output_mode"] = loaded.output_mode
        st.session_state["custom_structured"] = loaded.structured_output
        st.session_state["custom_reasoning"] = loaded.reasoning_effort or ""
        st.rerun()

    st.text_input("Prompt name", key="custom_name")
    st.text_area("System prompt", height=120, key="custom_system")
    st.text_area("User prompt template", height=260, key="custom_user")
    a, b, c = st.columns(3)
    a.selectbox("Output mode", ["label", "json"], key="custom_output_mode")
    b.checkbox("Structured Output / schema", key="custom_structured")
    c.selectbox("Reasoning effort", ["", "low", "medium", "high"], key="custom_reasoning")

    try:
        custom_strategy = build_custom_from_state()
        preview_ticket = st.text_area("Preview ticket", "Please cancel my subscription before the next renewal.", key="en_custom_preview_ticket")
        preview_payload = custom_strategy.build(preview_ticket)
        st.markdown("#### Rendered prompt")
        st.code(f"SYSTEM:\n{preview_payload.instructions or '(none)'}\n\nUSER:\n{preview_payload.input_text}", language="text")
        prompt_chars = len((preview_payload.instructions or "") + preview_payload.input_text)
        c1, c2, c3 = st.columns(3)
        c1.metric("Characters", f"{prompt_chars:,}")
        c2.metric("Approx. prompt tokens", f"{prompt_chars/4:.0f}")
        c3.metric("Structured", "Yes" if preview_payload.structured_output else "No")

        if st.button("💾 Save custom prompt", key="en_save_custom"):
            path = save_custom_prompt(custom_strategy)
            st.success(f"Saved: {path}")

        left, right = st.columns(2)
        with left:
            if st.button("▶ Run one ticket", type="primary", key="en_run_custom_single"):
                try:
                    client = build_client()
                    response = client.classify(preview_payload)
                    parsed = parse_prediction(response.raw_output, preview_payload.output_mode)
                    st.metric("Prediction", parsed.label or "__invalid__")
                    st.code(response.raw_output or "(empty response)", language="json" if preview_payload.output_mode == "json" else "text")
                    st.json({
                        "valid_output": parsed.valid_output,
                        "valid_json": parsed.valid_json,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "total_tokens": response.total_tokens,
                        "latency_seconds": response.latency_seconds,
                        "provider": response.provider,
                        "model": response.model,
                        "error": response.error,
                    })
                except Exception as exc:
                    st.error(f"Run failed: {type(exc).__name__}: {exc}")
        with right:
            custom_limit = st.number_input("Custom benchmark rows", 6, 300, 48, 6, key="en_custom_bench_limit")
            if st.button("🏁 Custom prompt mini benchmark", key="en_run_custom_bench"):
                try:
                    data = pd.read_csv(PATHS.processed_data / "benchmark.csv")
                    client = build_client()
                    inp, outp = get_provider_pricing(pricing, provider)
                    output_path = PATHS.results / "ui_runs" / provider / f"{custom_strategy.name}.csv"
                    result = run_strategy(data, custom_strategy, client, output_path, inp, outp, limit=int(custom_limit), force=True)
                    metrics = classification_metrics(result)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                    c2.metric("Macro F1", f"{metrics['macro_f1']:.4f}")
                    c3.metric("Tokens/request", f"{metrics['mean_total_tokens']:.0f}")
                    c4.metric("P95 latency", f"{metrics['p95_latency_seconds']:.3f}s")
                    st.plotly_chart(confusion_figure(result, "Custom prompt confusion matrix"), use_container_width=True)
                except Exception as exc:
                    st.error(f"Benchmark failed: {type(exc).__name__}: {exc}")
    except Exception as exc:
        st.warning(str(exc))


with tab_playground:
    render_playground_panel(
        language="en",
        provider=provider,
        model=model,
        strategies=STRATEGIES,
        strategy_title=strategy_title,
        build_client=build_client,
    )


with tab_benchmark:
    render_benchmark_panel(
        language="en",
        provider=provider,
        model=model,
        strategies=STRATEGIES,
        strategy_title=strategy_title,
        build_client=build_client,
        pricing_config=pricing,
    )


with tab_sweep:
    st.subheader("Decoding-parameter laboratory")
    st.write(
        "Prompt design and decoding are benchmarked separately. During a sweep, the prompt strategy is fixed while one decoding variable changes."
    )
    sweep_strategy = st.selectbox(
        "Fixed prompt strategy",
        STRATEGIES,
        index=STRATEGIES.index("p16_full_advanced_template"),
        format_func=strategy_title,
    )
    sweep_limit = st.slider("Rows per setting", 6, 120, 24, 6)
    supported = [name for name in ["temperature", "top_p", "top_k"] if caps.get(name)]
    st.write("Supported by selected provider adapter:", ", ".join(supported) if supported else "none")

    if st.button("Run parameter sweep", disabled=not supported):
        command = [
            sys.executable,
            "05_scripts/06_run_parameter_sweep.py",
            "--provider",
            provider,
            "--strategy",
            sweep_strategy,
            "--limit",
            str(sweep_limit),
            "--force",
        ]
        with st.spinner("Running parameter sweep..."):
            code, output = run_command(command)
        st.code(output or "(no console output)")
        if code == 0:
            st.success("Parameter sweep completed.")
            st.cache_data.clear()
        else:
            st.error("Parameter sweep failed.")

    sweep_path = PATHS.results / "parameter_sweeps" / provider / "parameter_sweep_summary.csv"
    sweep = safe_read_csv(sweep_path)
    if sweep is not None and not sweep.empty:
        metric = st.selectbox(
            "Sweep metric",
            ["macro_f1", "accuracy", "invalid_output_rate", "mean_total_tokens", "p95_latency_seconds"],
        )
        st.plotly_chart(parameter_sweep_figure(sweep, metric), use_container_width=True)
        st.dataframe(sweep, use_container_width=True, hide_index=True)
    else:
        st.info("No parameter-sweep result exists for this provider yet.")


with tab_validation:
    st.subheader("Output reliability and error analysis")
    summary = load_provider_summary(provider)
    available_raw = [s for s in STRATEGIES if load_provider_raw(provider, s) is not None]
    if not available_raw:
        st.info("Run at least one benchmark strategy first.")
    else:
        selected_validation = st.selectbox("Inspect raw strategy", available_raw, format_func=strategy_title)
        raw = load_provider_raw(provider, selected_validation)
        assert raw is not None

        m = classification_metrics(raw)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{m['accuracy']:.4f}")
        c2.metric("Macro F1", f"{m['macro_f1']:.4f}")
        c3.metric("Output valid", f"{m['output_contract_valid_rate']:.1%}")
        c4.metric("Invalid JSON", f"{m['invalid_json_rate']:.1%}" if pd.notna(m["invalid_json_rate"]) else "N/A")
        c5.metric("P95 latency", f"{m['p95_latency_seconds']:.3f}s")

        left, right = st.columns(2)
        with left:
            st.plotly_chart(confusion_figure(raw, f"Confusion matrix — {strategy_title(selected_validation)}"), use_container_width=True)
        with right:
            report = per_class_report(raw).reset_index().rename(columns={"index": "label", "f1-score": "f1"})
            st.plotly_chart(
                px.bar(report, x="label", y=["precision", "recall", "f1"], barmode="group", title="Per-class precision / recall / F1"),
                use_container_width=True,
            )

        invalid = raw[(~raw["valid_output"].astype(bool)) | (raw["predicted_label"] != raw["true_label"])].copy()
        st.markdown(f"#### Errors / invalid cases ({len(invalid)} of {len(raw)})")
        st.dataframe(
            invalid[[c for c in ["sample_id", "text", "true_label", "predicted_label", "raw_response", "valid_output", "valid_json", "error"] if c in invalid.columns]],
            use_container_width=True,
            hide_index=True,
        )

        raw_frames = {s: load_provider_raw(provider, s) for s in available_raw}
        raw_frames = {k: v for k, v in raw_frames.items() if v is not None}
        latency_box = token_latency_box(raw_frames, "latency_seconds")
        token_box = token_latency_box(raw_frames, "total_tokens")
        if latency_box is not None and token_box is not None:
            a, b = st.columns(2)
            a.plotly_chart(latency_box, use_container_width=True)
            b.plotly_chart(token_box, use_container_width=True)


with tab_finetune:
    st.subheader("Fine-tuning readiness")
    st.write(
        "The benchmark is intentionally completed before fine-tuning. This prevents a fine-tuned model from being evaluated on examples it has seen during training."
    )
    train_path = PATHS.data / "fine_tuning" / "sft_train.jsonl"
    val_path = PATHS.data / "fine_tuning" / "sft_validation.jsonl"
    c1, c2 = st.columns(2)
    c1.metric("SFT train exists", "Yes" if train_path.exists() else "No")
    c2.metric("SFT validation exists", "Yes" if val_path.exists() else "No")

    if st.button("Export leakage-safe SFT JSONL"):
        code, output = run_command([sys.executable, "05_scripts/12_export_finetuning_data.py"])
        st.code(output or "(no console output)")
        if code == 0:
            st.success("Fine-tuning dataset exported.")
        else:
            st.error("Export failed.")

    st.code(
        "python 05_scripts/04_run_benchmark.py --provider <provider> --model <fine-tuned-model-id> --strategy p16_full_advanced_template",
        language="bash",
    )
    st.caption(
        "Training is not auto-triggered because cloud fine-tuning can cost money, while local LoRA requirements depend on GPU memory and the selected base model."
    )


with tab_api:
    st.subheader("Provider-neutral API integration")
    st.write(
        "The benchmark runner uses one common client interface, so the same prompt strategy can be executed with Mock, local Ollama, Groq, Gemini or OpenAI. "
        "Real providers report their actual token usage when the SDK exposes it; latency is measured around the API request."
    )

    provider_table = pd.DataFrame(
        [
            {"provider": "mock", "key/env": "none", "where": "offline simulation", "tokens": "estimated", "latency": "simulated"},
            {"provider": "ollama", "key/env": "none", "where": "local machine", "tokens": "model/runtime reported", "latency": "wall-clock"},
            {"provider": "groq", "key/env": "GROQ_API_KEY", "where": "cloud API", "tokens": "provider reported", "latency": "wall-clock"},
            {"provider": "gemini", "key/env": "GEMINI_API_KEY", "where": "cloud API", "tokens": "provider reported", "latency": "wall-clock"},
            {"provider": "openai", "key/env": "OPENAI_API_KEY", "where": "cloud API", "tokens": "provider reported", "latency": "wall-clock"},
        ]
    )
    st.dataframe(provider_table, use_container_width=True, hide_index=True)

    st.markdown("#### Current runtime configuration")
    st.json(
        {
            "provider": provider,
            "model": model,
            "temperature": temperature if temperature_enabled else None,
            "top_p": top_p if top_p_enabled else None,
            "top_k": top_k if top_k_enabled else None,
            "capabilities": caps,
        }
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Test provider connection", key="api_connection_test"):
            command = [sys.executable, "05_scripts/03_check_provider.py", "--provider", provider]
            with st.spinner("Sending provider test request..."):
                code, output = run_command(command)
            st.code(output or "(no output)")
            if code == 0:
                st.success("Provider test completed successfully.")
            else:
                st.error("Provider test failed. Check API key, quota/model availability or local Ollama service.")
    with c2:
        st.code(
            "# .env examples\n"
            "GROQ_API_KEY=...\n"
            "GEMINI_API_KEY=...\n"
            "OPENAI_API_KEY=...\n"
            "OLLAMA_MODEL=llama3.2:3b",
            language="bash",
        )

    if provider == "gemini":
        st.markdown("#### Gemini-specific health check")
        st.caption("This calls the Google REST endpoint directly and never prints the API key. Enter the key in the shared sidebar GEMINI_API_KEY field.")
        if st.button("Test Gemini key + model", key="en_gemini_rest_test"):
            code, output = run_command([sys.executable, "05_scripts/15_test_gemini_connection.py", "--model", model], timeout=60)
            st.code(output or "(no output)")
            st.success("Gemini REST connection OK.") if code == 0 else st.error("Gemini test failed. Check key, network, quota and model ID.")

    st.markdown("#### Live provider request with full telemetry")
    api_strategy = st.selectbox("Prompt strategy", STRATEGIES, index=STRATEGIES.index("p16_full_advanced_template"), format_func=strategy_title, key="api_strategy")
    api_ticket = st.text_area(
        "Ticket",
        "I was charged twice last week. The billing issue is already resolved; my current request is to cancel before the next renewal.",
        key="api_ticket",
    )
    api_expected = st.selectbox("Expected label (optional)", ["(unknown)", *LABELS], key="api_expected")

    if st.button("Send live request", type="primary", key="api_live_request"):
        try:
            client = build_client()
            truth = api_expected if api_expected != "(unknown)" else "complaint"
            frame = pd.DataFrame([{"sample_id": "api_ui_1", "text": api_ticket, "true_label": truth}])
            out = PATHS.results / "ui_runs" / provider / "api_live.csv"
            inp, outp = get_provider_pricing(pricing, provider)
            result = run_strategy(frame, get_strategy(api_strategy), client, out, inp, outp, force=True)
            row = result.iloc[0]
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Prediction", str(row["predicted_label"]))
            c2.metric("Input tokens", int(row["input_tokens"]))
            c3.metric("Output tokens", int(row["output_tokens"]))
            c4.metric("Total tokens", int(row["total_tokens"]))
            c5.metric("Latency", f"{float(row['latency_seconds']):.3f}s")
            c6.metric("Estimated cost", f"${float(row['estimated_cost_usd']):.6f}")
            st.json(
                {
                    "provider": row["provider"],
                    "model": row["model"],
                    "token_source": row.get("token_source"),
                    "latency_source": row.get("latency_source"),
                    "valid_output": bool(row["valid_output"]),
                    "valid_json": None if pd.isna(row["valid_json"]) else bool(row["valid_json"]),
                    "temperature": row.get("temperature"),
                    "top_p": row.get("top_p"),
                    "top_k": row.get("top_k"),
                    "branch_count": int(row.get("branch_count", 1)),
                }
            )
            st.code(str(row["raw_response"]), language="json" if "json" in api_strategy or "structured" in api_strategy or "grammar" in api_strategy or api_strategy == "p16_full_advanced_template" else "text")
            if api_expected != "(unknown)":
                st.success("Matches expected label") if row["predicted_label"] == api_expected else st.error(f"Expected {api_expected}, got {row['predicted_label']}")
        except Exception as exc:
            st.exception(exc)

    st.markdown("#### Minimal integration from your own Python code")
    integration_example = (
        "from dotenv import load_dotenv\n"
        "from prompt_benchmark.config import load_yaml\n"
        "from prompt_benchmark.llm.factory import create_llm_client\n"
        "from prompt_benchmark.prompts import get_strategy\n\n"
        "load_dotenv()\n"
        "cfg = load_yaml('configs/benchmark.yaml')\n"
        "client = create_llm_client('groq', cfg, overrides={'model': '<your-model>'})\n"
        "payload = get_strategy('p16_full_advanced_template').build('Your support ticket')\n"
        "response = client.classify(payload)\n"
        "print(response.raw_output, response.input_tokens, response.output_tokens, response.latency_seconds)\n"
    )
    st.code(integration_example, language="python")
    st.caption("For full evaluation, use run_strategy()/04_run_benchmark.py so parsing, validity, cost and checkpoint telemetry are logged consistently.")


with tab_history:
    render_history_panel("en", provider, strategy_title)


with tab_system:
    st.subheader("Environment and project status")
    st.write("Use this page to verify that the one-click Windows setup prepared everything the UI needs.")

    status_rows = []
    checks: Iterable[tuple[str, bool, str]] = [
        ("Virtual environment", (PROJECT_ROOT / ".venv").exists(), ".venv"),
        ("Mock raw data", (PATHS.mock_data / "mock_support_tickets.csv").exists(), "01_data/mock/mock_support_tickets.csv"),
        ("Development split", (PATHS.processed_data / "development.csv").exists(), "01_data/processed/development.csv"),
        ("Benchmark split", (PATHS.processed_data / "benchmark.csv").exists(), "01_data/processed/benchmark.csv"),
        ("Prompt template", (PATHS.prompts / "templates" / "full_prompt_template.yaml").exists(), "configs/prompts/templates/full_prompt_template.yaml"),
        ("Results directory", PATHS.results.exists(), "07_outputs/results"),
        ("Reports directory", PATHS.reports.exists(), "07_outputs/reports"),
    ]
    for name, ok, path in checks:
        status_rows.append({"component": name, "status": "OK" if ok else "MISSING", "path": path})
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Dependency check")
    if st.button("Check installed package versions"):
        code, output = run_command([sys.executable, "00_setup/00_dependency_manager.py", "--check-only"])
        st.code(output or "(no output)")
        if code == 0:
            st.success("All requirements are satisfied and pip check passed.")
        else:
            st.warning("One or more dependencies need repair. Run 00_setup/05_update_environment.bat on Windows.")

    st.markdown("#### Provider connection test")
    if st.button("Test selected provider"):
        command = [sys.executable, "05_scripts/03_check_provider.py", "--provider", provider]
        with st.spinner("Testing provider..."):
            code, output = run_command(command)
        st.code(output or "(no output)")
        if code == 0:
            st.success("Provider test completed successfully.")
        else:
            st.error("Provider test failed. Check model availability, local Ollama status or API key/quota.")

    st.markdown("#### Windows launchers")
    st.code(
        "RUN_UI.bat\n"
        "00_setup\\01_setup_windows.bat\n"
        "00_setup\\02_run_ui.bat\n"
        "00_setup\\05_update_environment.bat",
        language="text",
    )

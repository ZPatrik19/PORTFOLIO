"""Generate reproducible project-level statistics and diagnostic plots.

The report intentionally separates four questions:
1) How large is the project and its data layer?
2) How diverse and balanced are the synthetic datasets?
3) How well do the routing/orchestration methodologies perform?
4) How expensive/complex are the benchmark queries and tool workflows?

The script never trains or modifies the model. It only reads repository artifacts and
writes descriptive CSV/JSON/PNG outputs under ``06_results/project_statistics``.
"""
from __future__ import annotations

import ast
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "03_src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from travel_agent.presets import PRESET_QUESTIONS
from travel_agent.tools import ToolRegistry

RAW = ROOT / "01_data" / "raw"
BENCH = ROOT / "01_data" / "benchmark" / "agent_tasks.json"
OUT = ROOT / "06_results" / "project_statistics"
OUT.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path, default=None):
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_q(series: pd.Series, q: float):
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.quantile(q)) if len(numeric) else None


def dataset_statistics() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW.glob("*.csv")):
        df = pd.read_csv(path)
        rows.append({
            "dataset": path.name,
            "rows": len(df),
            "columns": len(df.columns),
            "exact_duplicate_rows": int(df.duplicated().sum()),
            "missing_cells": int(df.isna().sum().sum()),
            "missing_pct": round(float(df.isna().sum().sum()) / max(1, df.size), 6),
            "memory_mb": round(float(df.memory_usage(deep=True).sum()) / (1024 * 1024), 3),
            "unique_cities": int(df["city"].nunique()) if "city" in df.columns else None,
        })
    return pd.DataFrame(rows)


def query_statistics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Reuse normalized-pattern metrics from the dedicated quality audit. Recomputing
    # them here would make a simple statistics refresh unnecessarily expensive on
    # the 366k-query corpus.
    quality = _load_json(ROOT / "06_results" / "data_quality" / "data_quality_report.json")
    audited = quality.get("query_datasets", {})
    summaries = []
    tool_counter: Counter[str] = Counter()
    complexity_counter: Counter[int] = Counter()
    for name in ["sample_user_queries.csv", "intent_router_dataset.csv", "intent_router_challenge.csv"]:
        path = RAW / name
        df = pd.read_csv(path)
        q = df["query"].astype(str)
        label_counts = df["tool_labels"].fillna("").astype(str).map(lambda s: len([x for x in s.split("|") if x]))
        if name == "intent_router_dataset.csv":
            for labels in df["tool_labels"].fillna("").astype(str):
                tool_counter.update(x for x in labels.split("|") if x)
            complexity_counter.update(int(v) for v in label_counts)
        aq = audited.get(name, {})
        summaries.append({
            "dataset": name,
            "rows": len(df),
            "unique_queries": int(q.nunique()),
            "exact_duplicate_queries": int(q.duplicated().sum()),
            "normalized_patterns": int(aq.get("normalized_patterns", 0)),
            "normalized_pattern_ratio": round(float(aq.get("normalized_pattern_ratio", 0)), 6),
            "largest_pattern_share": round(float(aq.get("largest_pattern_share", 0)), 6),
            "avg_query_chars": round(float(q.str.len().mean()), 2),
            "median_query_chars": round(float(q.str.len().median()), 2),
            "p95_query_chars": round(float(q.str.len().quantile(.95)), 2),
            "avg_intents_per_query": round(float(label_counts.mean()), 3),
            "multi_intent_rate": round(float((label_counts > 1).mean()), 6),
            "hu_share": round(float((df["language"].astype(str) == "hu").mean()), 6) if "language" in df.columns else None,
        })
    labels_df = pd.DataFrame([{"intent": k, "count": v} for k, v in sorted(tool_counter.items())])
    complexity_df = pd.DataFrame([{"intent_count": k, "queries": v} for k, v in sorted(complexity_counter.items())])
    return pd.DataFrame(summaries), labels_df, complexity_df


def entity_statistics() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    coverage = []
    configs = {
        "hotels.csv": ("name", "nightly_eur", "rating"),
        "restaurants.csv": ("name", "avg_meal_eur", "rating"),
        "attractions.csv": ("name", "ticket_eur", "rating"),
    }
    for name, (name_col, price_col, rating_col) in configs.items():
        df = pd.read_csv(RAW / name)
        names = df[name_col].astype(str)
        city_counts = df.groupby("city").size()
        rows.append({
            "dataset": name,
            "rows": len(df),
            "unique_names": int(names.nunique()),
            "unique_name_ratio": round(float(names.nunique() / max(1, len(df))), 6),
            "duplicate_names": int(names.duplicated().sum()),
            "cities": int(df["city"].nunique()),
            "rows_per_city_min": int(city_counts.min()),
            "rows_per_city_median": float(city_counts.median()),
            "rows_per_city_max": int(city_counts.max()),
            "rating_mean": round(float(pd.to_numeric(df[rating_col], errors="coerce").mean()), 3),
            "rating_p10": _safe_q(df[rating_col], .10),
            "rating_p50": _safe_q(df[rating_col], .50),
            "rating_p90": _safe_q(df[rating_col], .90),
            "price_p10": _safe_q(df[price_col], .10),
            "price_p50": _safe_q(df[price_col], .50),
            "price_p90": _safe_q(df[price_col], .90),
        })
        for city, count in city_counts.items():
            coverage.append({"dataset": name, "city": city, "rows": int(count)})
    return pd.DataFrame(rows), pd.DataFrame(coverage)


def benchmark_statistics() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    cases = _load_json(BENCH, default=[])
    complexity = Counter()
    tools = Counter()
    languages = Counter()
    qlength = []
    for case in cases:
        calls = case.get("expected_calls", [])
        complexity[len(calls)] += 1
        languages[str(case.get("language", "unknown"))] += 1
        qlength.append(len(str(case.get("query", ""))))
        tools.update(str(c.get("name")) for c in calls if c.get("name"))
    summary = {
        "cases": len(cases),
        "avg_expected_tool_calls": round(sum(k * v for k, v in complexity.items()) / max(1, len(cases)), 3),
        "multi_tool_rate": round(sum(v for k, v in complexity.items() if k > 1) / max(1, len(cases)), 6),
        "max_expected_tool_calls": max(complexity, default=0),
        "avg_query_chars": round(sum(qlength) / max(1, len(qlength)), 2),
        "median_query_chars": float(median(qlength)) if qlength else 0,
        "language_counts": dict(languages),
    }
    complexity_df = pd.DataFrame([{"tool_calls": k, "cases": v} for k, v in sorted(complexity.items())])
    tools_df = pd.DataFrame([{"tool": k, "expected_calls": v} for k, v in tools.most_common()])
    return summary, complexity_df, tools_df


def evaluation_statistics() -> pd.DataFrame:
    rows = []
    for methodology in ["rule_based", "plan_execute", "ml_router"]:
        metrics = _load_json(ROOT / "06_results" / methodology / "metrics.json")
        if metrics:
            rows.append({"methodology": methodology, **metrics})
    return pd.DataFrame(rows)


def repository_statistics() -> dict:
    source_files = list((ROOT / "03_src").rglob("*.py"))
    test_files = list((ROOT / "05_tests").glob("test_*.py"))
    docs = list((ROOT / "07_docs").rglob("*.md"))
    source_lines = sum(len(path.read_text(encoding="utf-8").splitlines()) for path in source_files)
    test_functions = 0
    for path in test_files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        test_functions += sum(1 for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"))
    notebooks = list((ROOT / "02_notebooks").glob("*.ipynb"))
    scripts = list((ROOT / "04_scripts").glob("*.py"))
    registry = ToolRegistry()
    bench = _load_json(BENCH, default=[])
    return {
        "registered_tools": len(registry.names), "tool_names": registry.names,
        "preset_scenarios": len(PRESET_QUESTIONS), "bilingual_preset_surface_forms": len(PRESET_QUESTIONS) * 2,
        "benchmark_cases": len(bench), "python_source_files": len(source_files), "python_source_lines": source_lines,
        "test_files": len(test_files), "test_functions": test_functions, "documentation_markdown_files": len(docs),
        "notebooks": len(notebooks), "runnable_scripts": len(scripts),
    }


STATIC_PALETTE = ["#2563EB", "#14B8A6", "#8B5CF6", "#F59E0B", "#EF4444", "#06B6D4", "#84CC16", "#EC4899"]


def _pretty(value: str) -> str:
    return str(value).replace(".csv", "").replace("_", " ").title()


def _compact_number(value: float) -> str:
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value/1_000:.1f}k"
    if abs(value) >= 10:
        return f"{value:,.0f}"
    return f"{value:.2f}"


def _style_axes(ax) -> None:
    # Static snapshots intentionally use a fixed high-contrast light theme so
    # they remain readable regardless of the Streamlit application's theme.
    ax.set_facecolor("#FFFFFF")
    ax.figure.set_facecolor("#FFFFFF")
    ax.tick_params(axis="both", colors="#111111", labelcolor="#111111")
    ax.xaxis.label.set_color("#111111")
    ax.yaxis.label.set_color("#111111")
    ax.title.set_color("#111111")
    for spine in ax.spines.values():
        spine.set_color("#4B5563")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D1D5DB", alpha=.55)
    ax.grid(axis="x", color="#D1D5DB", alpha=.55)
    ax.set_axisbelow(True)


def _barh(df: pd.DataFrame, label_col: str, value_col: str, title: str, xlabel: str, path: str, *, log_x: bool = False) -> None:
    if df.empty: return
    work = df.sort_values(value_col, ascending=True).copy()
    labels = work[label_col].astype(str).map(_pretty)
    values = work[value_col].astype(float)
    fig, ax = plt.subplots(figsize=(12, 7.2))
    colors = [STATIC_PALETTE[i % len(STATIC_PALETTE)] for i in range(len(work))]
    bars = ax.barh(labels, values, color=colors, alpha=.9)
    if log_x:
        ax.set_xscale("log")
    ax.bar_label(bars, labels=[_compact_number(v) for v in values], padding=5, fontsize=8)
    ax.set_title(title, loc="left", fontsize=17, fontweight="bold", pad=16)
    ax.set_xlabel(xlabel); ax.set_ylabel("")
    _style_axes(ax)
    fig.tight_layout(); fig.savefig(OUT / path, dpi=190, facecolor="#FFFFFF"); plt.close(fig)


def _bar(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, path: str, *, percent: bool = False) -> None:
    if df.empty: return
    work = df.copy()
    labels = work[x].astype(str).map(_pretty)
    values = work[y].astype(float)
    fig, ax = plt.subplots(figsize=(12, 7.2))
    colors = [STATIC_PALETTE[i % len(STATIC_PALETTE)] for i in range(len(work))]
    bars = ax.bar(labels, values, color=colors, alpha=.9)
    fmt = "{:.1%}" if percent else "{:,.3g}"
    ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=4, fontsize=9)
    ax.set_title(title, loc="left", fontsize=17, fontweight="bold", pad=16)
    ax.set_ylabel(ylabel); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=20)
    if percent:
        ax.set_ylim(0, 1.08); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    _style_axes(ax)
    fig.tight_layout(); fig.savefig(OUT / path, dpi=190, facecolor="#FFFFFF"); plt.close(fig)


def plot_methodology_quality(df: pd.DataFrame) -> None:
    wanted = ["tool_selection_accuracy", "tool_f1", "argument_accuracy", "task_success"]
    if df.empty or not all(c in df.columns for c in wanted): return
    chart = df.set_index("methodology")[wanted].rename(index={"rule_based":"Rule-based", "plan_execute":"Plan → Execute", "ml_router":"ML Router"})
    chart = chart.rename(columns={"tool_selection_accuracy":"Exact tool selection", "tool_f1":"Tool F1", "argument_accuracy":"Argument accuracy", "task_success":"Task success"})
    ax = chart.plot(kind="bar", figsize=(12, 7.2), ylim=(0, 1.05), color=STATIC_PALETTE[:4], width=.78)
    ax.set_title("Methodology quality benchmark", loc="left", fontsize=17, fontweight="bold", pad=16)
    ax.set_ylabel("Score"); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=0); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    _style_axes(ax); ax.legend(frameon=False, ncol=2)
    fig = ax.get_figure(); fig.tight_layout(); fig.savefig(OUT / "methodology_quality.png", dpi=190, facecolor="#FFFFFF"); plt.close(fig)


def plot_methodology_latency(df: pd.DataFrame) -> None:
    candidates = [c for c in ["latency_ms", "mean_latency_ms", "avg_latency_ms", "p95_latency_ms"] if c in df.columns]
    if df.empty or not candidates: return
    metric = candidates[0]
    work = df.copy(); work["methodology"] = work["methodology"].map({"rule_based":"Rule-based", "plan_execute":"Plan → Execute", "ml_router":"ML Router"}).fillna(work["methodology"])
    _bar(work, "methodology", metric, "Methodology mean runtime latency", "Milliseconds", "methodology_latency.png")


def plot_entity_diversity(entity_df: pd.DataFrame) -> None:
    _bar(entity_df, "dataset", "unique_name_ratio", "Entity-name diversity", "Unique-name ratio", "entity_name_diversity.png", percent=True)


def plot_query_diversity(query_df: pd.DataFrame) -> None:
    _bar(query_df, "dataset", "normalized_pattern_ratio", "Language-corpus diversity", "Normalized unique-pattern ratio", "query_diversity.png", percent=True)


def plot_inventory_per_city(coverage: pd.DataFrame) -> None:
    if coverage.empty: return
    pivot = coverage.pivot(index="city", columns="dataset", values="rows").sort_index()
    fig, ax = plt.subplots(figsize=(12, 7.2))
    for idx, col in enumerate(pivot.columns):
        ax.plot(range(len(pivot)), pivot[col], marker="o", markersize=3, linewidth=2, label=_pretty(col), color=STATIC_PALETTE[idx % len(STATIC_PALETTE)])
    ax.set_title("Inventory coverage by destination", loc="left", fontsize=17, fontweight="bold", pad=16)
    ax.set_xlabel("Destinations (alphabetical)"); ax.set_ylabel("Rows per city"); ax.set_xticks([]); ax.legend(frameon=False, ncol=3)
    _style_axes(ax)
    fig.tight_layout(); fig.savefig(OUT / "inventory_city_coverage.png", dpi=190, facecolor="#FFFFFF"); plt.close(fig)




def categorical_statistics() -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    specs = {
        "restaurant_cuisine": ("restaurants.csv", "cuisine"),
        "attraction_category": ("attractions.csv", "category"),
        "weather_condition": ("weather_fallback.csv", "condition"),
    }
    for key, (name, col) in specs.items():
        df = pd.read_csv(RAW / name, usecols=[col])
        counts = df[col].astype(str).value_counts().rename_axis("category").reset_index(name="count")
        outputs[key] = counts
    return outputs


def model_per_label_statistics(model: dict) -> pd.DataFrame:
    rows = []
    for label, metrics in (model.get("per_label") or {}).items():
        rows.append({"label": label, **metrics})
    return pd.DataFrame(rows)


def plot_numeric_distribution(filename: str, column: str, title: str, xlabel: str, output: str) -> None:
    path = RAW / filename
    df = pd.read_csv(path, usecols=[column])
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty: return
    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.hist(values, bins=36, color=STATIC_PALETTE[0], alpha=.86, edgecolor="white", linewidth=.6)
    ax.axvline(values.median(), color=STATIC_PALETTE[4], linestyle="--", linewidth=2, label=f"Median: {values.median():.1f}")
    ax.set_title(title, loc="left", fontsize=17, fontweight="bold", pad=16); ax.set_xlabel(xlabel); ax.set_ylabel("Records")
    _style_axes(ax); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / output, dpi=190, facecolor="#FFFFFF"); plt.close(fig)


def plot_model_per_label(df: pd.DataFrame) -> None:
    if df.empty or not {"label", "precision", "recall", "f1"}.issubset(df.columns): return
    chart = df.set_index("label")[["precision", "recall", "f1"]]
    ax = chart.plot(kind="bar", figsize=(12, 7.2), ylim=(0, 1.05), color=[STATIC_PALETTE[0], STATIC_PALETTE[1], STATIC_PALETTE[2]], width=.8)
    ax.set_title("Router performance by intent label", loc="left", fontsize=17, fontweight="bold", pad=16); ax.set_ylabel("Score"); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=20); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    _style_axes(ax); ax.legend(frameon=False, ncol=3)
    fig = ax.get_figure(); fig.tight_layout(); fig.savefig(OUT / "router_per_label_metrics.png", dpi=190, facecolor="#FFFFFF"); plt.close(fig)

def main() -> None:
    datasets = dataset_statistics()
    query_df, labels_df, query_complexity_df = query_statistics()
    entity_df, coverage_df = entity_statistics()
    bench_summary, bench_complexity_df, bench_tools_df = benchmark_statistics()
    evaluations = evaluation_statistics()
    repo = repository_statistics()
    model = _load_json(ROOT / "06_results" / "models" / "intent_router_metrics.json")
    per_label_df = model_per_label_statistics(model)
    categorical = categorical_statistics()
    quality = _load_json(ROOT / "06_results" / "data_quality" / "data_quality_report.json")
    presets = _load_json(ROOT / "06_results" / "preset_validation" / "preset_validation_summary.json")

    datasets.to_csv(OUT / "dataset_statistics.csv", index=False)
    query_df.to_csv(OUT / "query_statistics.csv", index=False)
    labels_df.to_csv(OUT / "tool_label_distribution.csv", index=False)
    query_complexity_df.to_csv(OUT / "query_complexity_distribution.csv", index=False)
    entity_df.to_csv(OUT / "entity_statistics.csv", index=False)
    coverage_df.to_csv(OUT / "city_inventory_coverage.csv", index=False)
    bench_complexity_df.to_csv(OUT / "benchmark_complexity.csv", index=False)
    bench_tools_df.to_csv(OUT / "benchmark_expected_tool_distribution.csv", index=False)
    evaluations.to_csv(OUT / "evaluation_statistics.csv", index=False)
    per_label_df.to_csv(OUT / "router_per_label_metrics.csv", index=False)
    for key, df in categorical.items():
        df.to_csv(OUT / f"{key}_distribution.csv", index=False)

    _barh(datasets, "dataset", "rows", "Dataset scale — row counts", "Rows (log scale)", "dataset_row_counts.png", log_x=True)
    _barh(datasets, "dataset", "memory_mb", "Dataset memory footprint", "Approx. MB (log scale)", "dataset_memory.png", log_x=True)
    _barh(labels_df, "intent", "count", "Intent-label distribution in router corpus", "Label occurrences", "intent_label_distribution.png")
    _bar(query_complexity_df, "intent_count", "queries", "Router query complexity", "Queries", "query_complexity.png")
    _bar(bench_complexity_df, "tool_calls", "cases", "Benchmark workflow complexity", "Cases", "benchmark_complexity.png")
    _barh(bench_tools_df, "tool", "expected_calls", "Expected tool usage in benchmark", "Expected calls", "benchmark_tool_distribution.png")
    plot_entity_diversity(entity_df)
    plot_query_diversity(query_df)
    plot_inventory_per_city(coverage_df)
    plot_methodology_quality(evaluations)
    plot_methodology_latency(evaluations)
    plot_model_per_label(per_label_df)
    for key, df in categorical.items():
        _barh(df, "category", "count", key.replace("_", " ").title(), "Records", f"{key}_distribution.png")
    plot_numeric_distribution("hotels.csv", "nightly_eur", "Hotel nightly-price distribution", "EUR / night", "hotel_price_distribution.png")
    plot_numeric_distribution("restaurants.csv", "avg_meal_eur", "Restaurant meal-price distribution", "EUR / person", "restaurant_price_distribution.png")
    plot_numeric_distribution("attractions.csv", "ticket_eur", "Attraction ticket-price distribution", "EUR", "attraction_ticket_distribution.png")

    total_rows = int(datasets["rows"].sum()) if not datasets.empty else 0
    summary = {
        "repository": repo,
        "data": {
            "raw_dataset_files": int(len(datasets)), "total_raw_rows": total_rows,
            "total_raw_memory_mb": round(float(datasets["memory_mb"].sum()), 3) if not datasets.empty else 0.0,
            "total_exact_duplicate_rows": int(datasets["exact_duplicate_rows"].sum()) if not datasets.empty else 0,
            "structured_inventory_rows": int(datasets[datasets["dataset"].isin(["hotels.csv", "attractions.csv", "restaurants.csv", "weather_fallback.csv"])] ["rows"].sum()),
        },
        "query_corpus": query_df.to_dict(orient="records"),
        "entity_inventory": entity_df.to_dict(orient="records"),
        "benchmark": bench_summary,
        "intent_router": {k: model.get(k) for k in ["train_rows","validation_rows","test_rows","validation_micro_f1","validation_macro_f1","test_micro_f1","test_macro_f1","test_hamming_loss","feature_model","thresholds","training_runtime"] if k in model},
        "data_quality": {"quality_gate_pass_rate": quality.get("quality_gate_pass_rate"), "quality_gates": quality.get("quality_gates", {}), "split_leakage": quality.get("split_leakage", {})},
        "preset_validation": presets,
        "evaluation": evaluations.to_dict(orient="records"),
    }
    (OUT / "project_statistics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

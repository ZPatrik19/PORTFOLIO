from __future__ import annotations

from pathlib import Path


import argparse
import os
import shutil

import pandas as pd
from dotenv import load_dotenv

from prompt_benchmark.paths import PATHS
from prompt_benchmark.utils.logging import configure_logging
from prompt_benchmark.config import load_yaml
from prompt_benchmark.evaluation.bootstrap import bootstrap_macro_f1_ci
from prompt_benchmark.evaluation.metrics import classification_metrics, grouped_classification_metrics, per_class_report
from prompt_benchmark.evaluation.plots import (
    bar_metric,
    plot_class_distribution,
    plot_cost_token_relationship,
    plot_case_type_accuracy_heatmap,
    plot_difficulty_accuracy,
    plot_distribution_box,
    plot_error_rate_by_class,
    plot_fix_regression_counts,
    plot_macro_f1_confidence_intervals,
    plot_per_class_f1_heatmap,
    plot_prompt_overhead_vs_f1,
    plot_relative_f1_improvement,
    plot_text_length_distribution,
    save_confusion_matrix,
    scatter_metric,
)
from prompt_benchmark.llm.factory import SUPPORTED_PROVIDERS
from prompt_benchmark.prompts import TECHNIQUE_CATALOG, get_strategy, list_strategies


def _prompt_stats() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    placeholder = "[CUSTOMER SUPPORT TICKET]"
    for strategy_name in list_strategies():
        strategy = get_strategy(strategy_name)
        payload = strategy.build(placeholder)
        instructions = payload.instructions or ""
        full_text = f"{instructions}\n{payload.input_text}"
        meta = TECHNIQUE_CATALOG[strategy_name]
        branch_count = len(strategy.build_branches(placeholder)) if hasattr(strategy, "build_branches") else 1
        rows.append(
            {
                "strategy": strategy_name,
                "strategy_title": meta["title"],
                "category": meta["category"],
                "hypothesis": meta["hypothesis"],
                "prompt_characters": len(full_text),
                "prompt_words": len(full_text.split()),
                "approx_prompt_tokens": len(full_text) / 4.0,
                "system_characters": len(instructions),
                "structured_output": payload.structured_output,
                "output_mode": payload.output_mode,
                "reasoning_effort": payload.reasoning_effort,
                "branch_count": branch_count,
                "system_prompt": instructions,
                "user_prompt_template": payload.input_text,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    configure_logging()
    load_dotenv()
    parser = argparse.ArgumentParser(description="Generate figures and aggregate metrics for one provider.")
    parser.add_argument(
        "--provider",
        choices=list(SUPPORTED_PROVIDERS),
        default=os.getenv("LLM_PROVIDER", "mock"),
    )
    args = parser.parse_args()

    raw_dir = PATHS.results / "raw" / args.provider
    raw_paths = sorted(raw_dir.glob("p*.csv"))
    if not raw_paths:
        raise FileNotFoundError(
            f"No raw benchmark files for provider '{args.provider}'. "
            f"Run 05_scripts/04_run_benchmark.py --provider {args.provider} first."
        )

    cfg = load_yaml(PATHS.configs / "benchmark.yaml")
    rows: list[dict[str, object]] = []
    frames: dict[str, pd.DataFrame] = {}
    per_class_reports: dict[str, pd.DataFrame] = {}

    reports_dir = PATHS.reports
    per_class_dir = reports_dir / "per_class" / args.provider
    fig_dir = reports_dir / "figures" / args.provider
    result_dir = PATHS.results / args.provider
    per_class_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    for path in raw_paths:
        frame = pd.read_csv(path)
        if frame.empty or "prompt_strategy" not in frame.columns:
            continue
        strategy = str(frame["prompt_strategy"].iloc[0])
        if strategy.startswith("full") or strategy.startswith("minus_"):
            continue

        frames[strategy] = frame
        metrics = classification_metrics(frame)
        ci_low, ci_high = bootstrap_macro_f1_ci(
            frame,
            int(cfg.get("bootstrap_iterations", 1000)),
            int(cfg.get("random_seed", 42)),
        )
        metrics.update(
            {
                "provider": str(frame.get("provider", pd.Series([args.provider])).iloc[0]),
                "model": str(frame["model"].iloc[0]),
                "strategy": strategy,
                "temperature": frame["temperature"].dropna().iloc[0] if "temperature" in frame and frame["temperature"].notna().any() else None,
                "top_p": frame["top_p"].dropna().iloc[0] if "top_p" in frame and frame["top_p"].notna().any() else None,
                "top_k": frame["top_k"].dropna().iloc[0] if "top_k" in frame and frame["top_k"].notna().any() else None,
                "token_source": str(frame["token_source"].iloc[0]) if "token_source" in frame else "unknown",
                "latency_source": str(frame["latency_source"].iloc[0]) if "latency_source" in frame else "unknown",
                "macro_f1_ci_low": ci_low,
                "macro_f1_ci_high": ci_high,
            }
        )
        rows.append(metrics)
        report = per_class_report(frame)
        per_class_reports[strategy] = report
        report.to_csv(per_class_dir / f"{strategy}.csv")

    if not rows:
        raise ValueError(f"No valid prompt-strategy benchmark files found for provider '{args.provider}'.")

    summary = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)

    baseline_row = summary[summary["strategy"] == "p0_zero_shot"]
    if not baseline_row.empty:
        baseline = baseline_row.iloc[0]
        summary["absolute_f1_improvement_vs_p0"] = summary["macro_f1"] - float(baseline["macro_f1"])
        summary["relative_f1_improvement_vs_p0_pct"] = (
            summary["absolute_f1_improvement_vs_p0"] / max(float(baseline["macro_f1"]), 1e-12) * 100.0
        )
        summary["token_overhead_vs_p0_pct"] = (
            summary["mean_total_tokens"] / max(float(baseline["mean_total_tokens"]), 1e-12) - 1.0
        ) * 100.0
        summary["latency_overhead_vs_p0_pct"] = (
            summary["p95_latency_seconds"] / max(float(baseline["p95_latency_seconds"]), 1e-12) - 1.0
        ) * 100.0
        summary["invalid_output_reduction_vs_p0_pp"] = (
            float(baseline["invalid_output_rate"]) - summary["invalid_output_rate"]
        ) * 100.0
    summary.to_csv(result_dir / "benchmark_summary.csv", index=False)

    prompt_stats = _prompt_stats()
    prompt_stats.to_csv(result_dir / "prompt_complexity.csv", index=False)

    scenario_rows: list[pd.DataFrame] = []
    difficulty_rows: list[pd.DataFrame] = []
    for strategy, frame in frames.items():
        by_case = grouped_classification_metrics(frame, "case_type")
        if not by_case.empty:
            by_case.insert(0, "strategy", strategy)
            scenario_rows.append(by_case)
        by_difficulty = grouped_classification_metrics(frame, "difficulty")
        if not by_difficulty.empty:
            by_difficulty.insert(0, "strategy", strategy)
            difficulty_rows.append(by_difficulty)
    if scenario_rows:
        pd.concat(scenario_rows, ignore_index=True).to_csv(result_dir / "case_type_summary.csv", index=False)
    if difficulty_rows:
        pd.concat(difficulty_rows, ignore_index=True).to_csv(result_dir / "difficulty_summary.csv", index=False)

    benchmark_path = PATHS.processed_data / "benchmark.csv"
    if benchmark_path.exists():
        benchmark_frame = pd.read_csv(benchmark_path)
        plot_class_distribution(benchmark_frame, fig_dir / "01_class_distribution.png")
        plot_text_length_distribution(benchmark_frame, fig_dir / "02_text_length_distribution.png")

    bar_metric(summary, "macro_f1", f"Macro F1 by Prompt Strategy — {args.provider}", "Macro F1", fig_dir / "03_macro_f1_comparison.png")
    bar_metric(summary, "accuracy", f"Accuracy by Prompt Strategy — {args.provider}", "Accuracy", fig_dir / "04_accuracy_comparison.png")
    bar_metric(summary, "invalid_output_rate", f"Invalid Output Rate — {args.provider}", "Rate", fig_dir / "05_invalid_output_rate.png")
    bar_metric(summary, "mean_total_tokens", f"Mean Token Usage — {args.provider}", "Tokens / request", fig_dir / "06_token_usage.png")
    bar_metric(summary, "p95_latency_seconds", f"P95 Latency — {args.provider}", "Seconds", fig_dir / "07_latency_comparison.png")
    bar_metric(summary, "cost_per_1000_requests_usd", f"Estimated Cost per 1K Requests — {args.provider}", "USD", fig_dir / "08_cost_comparison.png")
    scatter_metric(summary, "cost_per_1000_requests_usd", "macro_f1", f"Quality vs Cost — {args.provider}", "USD / 1K requests", "Macro F1", fig_dir / "09_quality_vs_cost.png")
    scatter_metric(summary, "p95_latency_seconds", "macro_f1", f"Quality vs Latency — {args.provider}", "P95 latency (s)", "Macro F1", fig_dir / "10_quality_vs_latency.png")
    plot_macro_f1_confidence_intervals(summary, fig_dir / "11_macro_f1_bootstrap_ci.png")
    plot_per_class_f1_heatmap(per_class_reports, fig_dir / "12_per_class_f1_heatmap.png")
    plot_error_rate_by_class(frames, fig_dir / "13_error_rate_by_class.png")
    plot_distribution_box(frames, "latency_seconds", f"Latency Distribution — {args.provider}", "Latency (seconds)", fig_dir / "14_latency_distribution_boxplot.png")
    plot_distribution_box(frames, "total_tokens", f"Token Distribution — {args.provider}", "Total tokens / request", fig_dir / "15_token_distribution_boxplot.png")

    baseline = frames.get("p0_zero_shot")
    best_name = str(summary.iloc[0]["strategy"])
    if baseline is not None:
        save_confusion_matrix(baseline, f"Baseline confusion matrix — {args.provider}", fig_dir / "16_confusion_matrix_baseline.png")
        plot_relative_f1_improvement(summary, "p0_zero_shot", fig_dir / "18_relative_f1_improvement.png")
    save_confusion_matrix(frames[best_name], f"Best strategy: {best_name} — {args.provider}", fig_dir / "17_confusion_matrix_best.png")
    if baseline is not None and best_name != "p0_zero_shot":
        plot_fix_regression_counts(baseline, frames[best_name], best_name, fig_dir / "19_fixed_vs_regressed_samples.png")
    plot_prompt_overhead_vs_f1(prompt_stats, summary, fig_dir / "20_prompt_length_vs_f1.png")
    plot_cost_token_relationship(summary, fig_dir / "21_token_usage_vs_cost.png")
    plot_case_type_accuracy_heatmap(frames, fig_dir / "22_case_type_accuracy_heatmap.png")
    plot_difficulty_accuracy(frames, fig_dir / "23_difficulty_accuracy.png")

    merged: pd.DataFrame | None = None
    for strategy, frame in frames.items():
        cols = frame[["sample_id", "text", "true_label", "predicted_label"]].rename(
            columns={"predicted_label": strategy}
        )
        merged = cols if merged is None else merged.merge(cols[["sample_id", strategy]], on="sample_id", how="inner")
    if merged is not None:
        pred_cols = list(frames)
        merged["number_of_failed_strategies"] = sum(
            (merged[column] != merged["true_label"]).astype(int) for column in pred_cols
        )
        merged.sort_values("number_of_failed_strategies", ascending=False).to_csv(
            reports_dir / f"hard_examples_{args.provider}.csv", index=False
        )

    # Convenience copies: notebooks can always read the most recently generated report.
    shutil.copy2(result_dir / "benchmark_summary.csv", "07_outputs/results/benchmark_summary.csv")
    shutil.copy2(result_dir / "prompt_complexity.csv", "07_outputs/results/prompt_complexity.csv")

    print(summary.to_string(index=False))
    print(f"\nProvider: {args.provider}")
    print(f"Best quality strategy: {best_name}")
    print(f"Saved provider-specific report -> {result_dir} and {fig_dir}")


if __name__ == "__main__":
    main()

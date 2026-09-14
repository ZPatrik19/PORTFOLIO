from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from prompt_benchmark.constants import INVALID_LABEL, LABELS


def _save(fig, path: Path) -> None:
    """Save one portfolio-ready figure and close it to avoid notebook memory leaks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def bar_metric(summary: pd.DataFrame, metric: str, title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(summary["strategy"], summary[metric])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Prompt strategy")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def scatter_metric(summary: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(summary[x], summary[y], s=70)
    for _, row in summary.iterrows():
        ax.annotate(str(row["strategy"]), (row[x], row[y]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    _save(fig, path)


def save_confusion_matrix(frame: pd.DataFrame, title: str, path: Path) -> None:
    y_pred = frame["predicted_label"].fillna(INVALID_LABEL).astype(str)
    cm = confusion_matrix(frame["true_label"], y_pred, labels=list(LABELS))
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, xticks_rotation=35, colorbar=False)
    ax.set_title(title)
    _save(fig, path)


def plot_class_distribution(frame: pd.DataFrame, path: Path) -> None:
    counts = frame["true_label"].value_counts().reindex(LABELS, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(counts.index, counts.values)
    ax.set_title("Benchmark Class Distribution")
    ax.set_xlabel("Intent class")
    ax.set_ylabel("Number of samples")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(counts.values):
        ax.text(idx, value, str(int(value)), ha="center", va="bottom")
    _save(fig, path)


def plot_text_length_distribution(frame: pd.DataFrame, path: Path) -> None:
    lengths = frame["text"].astype(str).str.split().str.len()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(lengths, bins=min(25, max(5, int(np.sqrt(len(lengths))))))
    ax.axvline(float(lengths.median()), linestyle="--", label=f"Median = {lengths.median():.0f} words")
    ax.set_title("Support Ticket Length Distribution")
    ax.set_xlabel("Words per ticket")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def plot_macro_f1_confidence_intervals(summary: pd.DataFrame, path: Path) -> None:
    ordered = summary.sort_values("macro_f1", ascending=True).reset_index(drop=True)
    lower = ordered["macro_f1"] - ordered["macro_f1_ci_low"]
    upper = ordered["macro_f1_ci_high"] - ordered["macro_f1"]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(ordered["macro_f1"], ordered["strategy"], xerr=np.vstack([lower, upper]), fmt="o", capsize=4)
    ax.set_title("Macro F1 with 95% Bootstrap Confidence Intervals")
    ax.set_xlabel("Macro F1")
    ax.set_ylabel("Prompt strategy")
    ax.grid(axis="x", alpha=0.25)
    _save(fig, path)


def plot_per_class_f1_heatmap(per_class_reports: Mapping[str, pd.DataFrame], path: Path) -> None:
    strategies = list(per_class_reports)
    matrix = np.array([
        [float(per_class_reports[s].loc[label, "f1-score"]) for label in LABELS]
        for s in strategies
    ])
    fig, ax = plt.subplots(figsize=(10, max(5, len(strategies) * 0.55)))
    image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=30, ha="right")
    ax.set_yticks(range(len(strategies)), strategies)
    ax.set_title("Per-Class F1 Across Prompt Strategies")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="F1 score")
    _save(fig, path)


def plot_error_rate_by_class(frames: Mapping[str, pd.DataFrame], path: Path) -> None:
    rows: list[dict[str, object]] = []
    for strategy, frame in frames.items():
        for label in LABELS:
            subset = frame[frame["true_label"] == label]
            if subset.empty:
                continue
            rows.append({"strategy": strategy, "label": label, "error_rate": 1.0 - float(subset["correct"].astype(bool).mean())})
    data = pd.DataFrame(rows)
    if data.empty:
        return
    pivot = data.pivot(index="strategy", columns="label", values="error_rate").reindex(columns=LABELS)
    fig, ax = plt.subplots(figsize=(10, max(5, len(pivot) * 0.55)))
    image = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Classification Error Rate by True Class")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Error rate")
    _save(fig, path)


def plot_distribution_box(frames: Mapping[str, pd.DataFrame], column: str, title: str, ylabel: str, path: Path) -> None:
    labels, series = [], []
    for strategy, frame in frames.items():
        values = pd.to_numeric(frame[column], errors="coerce").dropna().values
        if len(values):
            labels.append(strategy)
            series.append(values)
    if not series:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.boxplot(series, tick_labels=labels, showfliers=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Prompt strategy")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def plot_relative_f1_improvement(summary: pd.DataFrame, baseline_strategy: str, path: Path) -> None:
    baseline_rows = summary[summary["strategy"] == baseline_strategy]
    if baseline_rows.empty:
        return
    baseline = float(baseline_rows.iloc[0]["macro_f1"])
    data = summary.copy()
    data["absolute_f1_improvement"] = data["macro_f1"] - baseline
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(data["strategy"], data["absolute_f1_improvement"])
    ax.axhline(0, linewidth=1)
    ax.set_title("Macro F1 Improvement Relative to Zero-Shot Baseline")
    ax.set_ylabel("Absolute Macro F1 change")
    ax.set_xlabel("Prompt strategy")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def plot_fix_regression_counts(baseline: pd.DataFrame, candidate: pd.DataFrame, candidate_name: str, path: Path) -> None:
    merged = baseline[["sample_id", "true_label", "predicted_label"]].merge(
        candidate[["sample_id", "predicted_label"]], on="sample_id", suffixes=("_baseline", "_candidate")
    )
    baseline_correct = merged["predicted_label_baseline"] == merged["true_label"]
    candidate_correct = merged["predicted_label_candidate"] == merged["true_label"]
    counts = pd.Series({
        "Correct in both": int((baseline_correct & candidate_correct).sum()),
        "Fixed by candidate": int((~baseline_correct & candidate_correct).sum()),
        "Regression": int((baseline_correct & ~candidate_correct).sum()),
        "Wrong in both": int((~baseline_correct & ~candidate_correct).sum()),
    })
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(counts.index, counts.values)
    ax.set_title(f"Baseline vs {candidate_name}: Sample-Level Outcome Changes")
    ax.set_ylabel("Number of benchmark samples")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(counts.values):
        ax.text(idx, value, str(int(value)), ha="center", va="bottom")
    _save(fig, path)


def plot_prompt_overhead_vs_f1(prompt_stats: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    merged = prompt_stats.merge(summary[["strategy", "macro_f1"]], on="strategy", how="inner")
    if merged.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(merged["approx_prompt_tokens"], merged["macro_f1"], s=70)
    for _, row in merged.iterrows():
        ax.annotate(str(row["strategy"]), (row["approx_prompt_tokens"], row["macro_f1"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_title("Approximate Prompt Length vs Macro F1")
    ax.set_xlabel("Approximate prompt tokens (character count / 4)")
    ax.set_ylabel("Macro F1")
    ax.grid(alpha=0.25)
    _save(fig, path)


def plot_cost_token_relationship(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(summary["mean_total_tokens"], summary["cost_per_1000_requests_usd"], s=70)
    for _, row in summary.iterrows():
        ax.annotate(str(row["strategy"]), (row["mean_total_tokens"], row["cost_per_1000_requests_usd"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_title("Token Usage vs Estimated API Cost")
    ax.set_xlabel("Mean tokens per request")
    ax.set_ylabel("Estimated USD / 1K requests")
    ax.grid(alpha=0.25)
    _save(fig, path)


def plot_case_type_accuracy_heatmap(frames: Mapping[str, pd.DataFrame], path: Path) -> None:
    rows: list[dict[str, object]] = []
    for strategy, frame in frames.items():
        if "case_type" not in frame.columns:
            continue
        for case_type, subset in frame.groupby("case_type"):
            rows.append({
                "strategy": strategy,
                "case_type": str(case_type),
                "accuracy": float(subset["correct"].astype(bool).mean()),
            })
    data = pd.DataFrame(rows)
    if data.empty:
        return
    pivot = data.pivot(index="strategy", columns="case_type", values="accuracy")
    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.55)))
    image = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Accuracy by Synthetic Scenario Type")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Accuracy")
    _save(fig, path)


def plot_difficulty_accuracy(frames: Mapping[str, pd.DataFrame], path: Path) -> None:
    rows: list[dict[str, object]] = []
    for strategy, frame in frames.items():
        if "difficulty" not in frame.columns:
            continue
        for difficulty, subset in frame.groupby("difficulty"):
            rows.append({
                "strategy": strategy,
                "difficulty": str(difficulty),
                "accuracy": float(subset["correct"].astype(bool).mean()),
            })
    data = pd.DataFrame(rows)
    if data.empty:
        return
    pivot = data.pivot(index="strategy", columns="difficulty", values="accuracy")
    desired = [c for c in ["easy", "medium", "hard"] if c in pivot.columns]
    pivot = pivot.reindex(columns=desired)
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(pivot.index))
    width = 0.8 / max(1, len(pivot.columns))
    for idx, difficulty in enumerate(pivot.columns):
        ax.bar(x + idx * width, pivot[difficulty].values, width, label=difficulty)
    ax.set_xticks(x + width * (len(pivot.columns) - 1) / 2, pivot.index, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Prompt strategy")
    ax.set_title("Prompt Robustness by Difficulty")
    ax.legend(title="Difficulty")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, path)

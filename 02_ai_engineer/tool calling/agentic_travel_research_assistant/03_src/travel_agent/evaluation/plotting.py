from __future__ import annotations

import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt


def _save(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(metrics_path: Path, output_path: Path) -> None:
    """Render the primary quality dashboard.

    Kept under the original function name so older scripts/notebooks remain compatible.
    """
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    labels = [
        "Tool selection\n(exact)",
        "Tool precision",
        "Tool recall",
        "Tool F1",
        "Argument\naccuracy",
        "Task success",
    ]
    values = [
        metrics["tool_selection_accuracy"],
        metrics["tool_precision"],
        metrics["tool_recall"],
        metrics["tool_f1"],
        metrics["argument_accuracy"],
        metrics["task_success"],
    ]

    fig, ax = plt.subplots(figsize=(11, 6.2))
    y = list(range(len(labels)))
    bars = ax.barh(y, values, color=["#2E6F9E", "#4B86B4", "#6AA6C8", "#6B5AA6", "#C96B2C", "#2E7D5B"])
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Score")
    method_label = output_path.parent.name.replace("_", " ").title()
    ax.set_title(f"Agent quality dashboard — {method_label}", pad=14, fontsize=15)
    ax.axvline(0.90, color="#9A9A9A", linestyle="--", linewidth=1.2)
    ax.text(0.905, 0.985, "0.90 reference", transform=ax.get_xaxis_transform(), color="#666666", fontsize=9, va="top")
    ax.grid(axis="x", alpha=0.16)
    for bar, value in zip(bars, values):
        ax.text(min(value + 0.012, 0.965), bar.get_y() + bar.get_height()/2, f"{value*100:.1f}%", va="center", fontsize=10, fontweight="bold")
    ax.text(
        0.0,
        -0.13,
        "Read together: exact selection penalizes extra/missing tools; P/R/F1 diagnose selection; argument accuracy measures parameter extraction; task success is strict end-to-end success.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        va="top",
    )
    _save(fig, output_path)


def plot_efficiency(metrics_path: Path, output_path: Path) -> None:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(10, 5.8))
    labels = ["Average tool calls\nper request", "Unnecessary\ntool-call rate"]
    values = [metrics["num_calls"], metrics["unnecessary_tool_call_rate"] * 100]
    units = ["calls", "%"]
    bars = ax.bar(labels, values, color=["#2E6F9E", "#C96B2C"], width=0.56)
    method_label = output_path.parent.name.replace("_", " ").title()
    ax.set_title(f"Orchestration efficiency — {method_label}", pad=14, fontsize=15)
    ax.set_ylabel("Value (different units shown on labels)")
    ax.grid(axis="y", alpha=0.16)
    maxv = max(values) if values else 1
    ax.set_ylim(0, maxv * 1.32 + 0.2)
    for bar, value, unit in zip(bars, values, units):
        label = f"{value:.2f} {unit}" if unit == "calls" else f"{value:.2f}%"
        ax.text(bar.get_x() + bar.get_width()/2, value + maxv*0.04, label, ha="center", va="bottom", fontweight="bold")
    ax.text(
        0.0,
        -0.14,
        "Lower is generally better here, provided task success remains high. Extra calls create latency/cost and can be risky for state-changing tools.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        va="top",
    )
    _save(fig, output_path)


def plot_latency(metrics_path: Path, output_path: Path) -> None:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    labels = ["Mean latency", "P95 latency"]
    values = [metrics["latency_ms"], metrics["p95_latency_ms"]]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    bars = ax.bar(labels, values, color=["#6B5AA6", "#B44545"], width=0.58)
    method_label = output_path.parent.name.replace("_", " ").title()
    ax.set_title(f"End-to-end latency — {method_label}", pad=14, fontsize=15)
    ax.set_ylabel("Milliseconds")
    ax.grid(axis="y", alpha=0.16)
    maxv = max(values) if values else 1
    ax.set_ylim(0, maxv * 1.35 + 0.01)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, value + maxv*0.05, f"{value:.3f} ms", ha="center", fontweight="bold")
    ax.text(
        0.0,
        -0.14,
        "Offline numbers mainly measure local Python overhead. In LLM mode the same metric includes network/model time and becomes operationally meaningful.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        va="top",
    )
    _save(fig, output_path)


def plot_case_diagnostics(case_metrics_path: Path, output_path: Path) -> None:
    rows = []
    with case_metrics_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    case_ids = [row["id"] for row in rows]
    metrics = ["tool_selection_accuracy", "argument_accuracy", "task_success"]
    labels = ["Tool selection", "Argument accuracy", "Task success"]
    matrix = [[float(row[m]) for row in rows] for m in metrics]

    fig, ax = plt.subplots(figsize=(13, 4.8))
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(range(len(case_ids)), case_ids, rotation=55, ha="right", fontsize=8)
    ax.set_title("Case-level diagnostic matrix — where does the baseline fail?", pad=14, fontsize=15)
    for i, metric_values in enumerate(matrix):
        for j, value in enumerate(metric_values):
            ax.text(j, i, f"{value:.2f}" if 0 < value < 1 else str(int(value)), ha="center", va="center", fontsize=7, color="#222222")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Score")
    ax.text(
        0.0,
        -0.32,
        "Use this plot after aggregate metrics: red/yellow cells identify exact cases worth opening in evaluation_details.json for root-cause analysis.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        va="top",
    )
    _save(fig, output_path)


def plot_methodology_comparison(comparison_path: Path, output_path: Path) -> None:
    """Compare core quality metrics across orchestration methodologies."""
    data = json.loads(comparison_path.read_text(encoding="utf-8"))
    methods = list(data.keys())
    metric_keys = ["tool_selection_accuracy", "argument_accuracy", "task_success", "tool_f1"]
    labels = ["Tool selection", "Argument accuracy", "Task success", "Tool F1"]

    fig, ax = plt.subplots(figsize=(11, 6.4))
    width = 0.34 if len(methods) <= 2 else 0.8 / len(methods)
    x = list(range(len(metric_keys)))
    for idx, method in enumerate(methods):
        offset = (idx - (len(methods) - 1) / 2) * width
        values = [data[method][key] for key in metric_keys]
        bars = ax.bar([i + offset for i in x], values, width=width, label=method)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value + 0.015, f"{value*100:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Methodology comparison — same dataset, different orchestration strategy", pad=14)
    ax.grid(axis="y", alpha=0.16)
    ax.legend()
    ax.text(
        0.0, -0.14,
        "Rule-based is the simple reproducible baseline. Plan-then-execute exposes an inspectable plan and uses stronger parsing for benchmark edge cases.",
        transform=ax.transAxes, fontsize=9, color="#555555", va="top",
    )
    _save(fig, output_path)

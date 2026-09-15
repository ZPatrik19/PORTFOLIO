"""Figure generation helpers for retrieval, corpus and monitoring diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .models import Chunk

FIGURE_DPI = 150


def _save_series_plot(
    series: pd.Series,
    kind: str,
    path: Path,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ylim: tuple[float, float] | None = None,
    title: str | None = None,
) -> None:
    """Render and persist a single pandas/matplotlib series plot."""

    figure = plt.figure()
    series.plot(kind=kind)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if ylim:
        plt.ylim(*ylim)
    if title:
        plt.title(title)
    plt.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)


def _save_latency_quality_scatter(method_summary: pd.DataFrame, output_path: Path) -> None:
    figure = plt.figure()
    plt.scatter(method_summary["P95 latency"], method_summary["Recall@5"])
    for method_name, row in method_summary.iterrows():
        plt.annotate(method_name, (row["P95 latency"], row["Recall@5"]))
    plt.xlabel("P95 latency (ms)")
    plt.ylabel("Recall@5")
    plt.tight_layout()
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)


def save_basic_figures(
    chunks: list[Chunk],
    quality_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    out_dir: Path,
    method_summary: pd.DataFrame | None = None,
    telemetry_df: pd.DataFrame | None = None,
) -> None:
    """Generate the standard portfolio plots from available deterministic artifacts."""

    del quality_df  # Reserved for richer quality plots without changing this public API.
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_frame = pd.DataFrame([chunk.model_dump() for chunk in chunks])

    if not chunk_frame.empty:
        _save_series_plot(
            chunk_frame.groupby("title").size().sort_values(),
            "barh",
            out_dir / "chunks_per_document.png",
            xlabel="Chunks",
        )
        _save_series_plot(
            chunk_frame["text"].str.len(),
            "hist",
            out_dir / "chunk_size_distribution.png",
            xlabel="Chunk size (characters)",
        )
        _save_series_plot(
            chunk_frame.groupby("language").size(),
            "bar",
            out_dir / "language_distribution.png",
            ylabel="Chunks",
        )
        _save_series_plot(
            chunk_frame.groupby("chunk_type").size(),
            "bar",
            out_dir / "chunk_type_distribution.png",
            ylabel="Chunks",
        )
        _save_series_plot(
            chunk_frame.groupby("source_type").size(),
            "bar",
            out_dir / "source_type_distribution.png",
            ylabel="Chunks",
        )
        topics = pd.Series(
            [keyword for keywords in chunk_frame["keywords"] for keyword in (keywords or [])],
            dtype="object",
        ).value_counts().head(15)
        if not topics.empty:
            _save_series_plot(
                topics.sort_values(),
                "barh",
                out_dir / "document_count_by_topic_proxy.png",
                xlabel="Chunk mentions",
            )

    if benchmark_df is not None and not benchmark_df.empty:
        _save_series_plot(
            benchmark_df[["recall@5", "mrr", "hit_rate"]].mean(),
            "bar",
            out_dir / "retrieval_metrics.png",
            ylim=(0, 1),
        )
        _save_series_plot(
            benchmark_df["latency_ms"],
            "hist",
            out_dir / "retrieval_latency.png",
            xlabel="Retrieval latency (ms)",
        )

    if method_summary is not None and not method_summary.empty:
        summary = method_summary.set_index("Method")
        _save_series_plot(summary["Recall@5"], "bar", out_dir / "recall_at_5_comparison.png", ylim=(0, 1))
        _save_series_plot(summary["MRR"], "bar", out_dir / "mrr_comparison.png", ylim=(0, 1))
        _save_series_plot(summary["P95 latency"], "bar", out_dir / "p95_latency_comparison.png", ylabel="ms")
        _save_latency_quality_scatter(summary, out_dir / "latency_vs_quality.png")

    if telemetry_df is None or telemetry_df.empty or "timestamp" not in telemetry_df:
        return

    telemetry = telemetry_df.copy()
    telemetry["timestamp"] = pd.to_datetime(telemetry["timestamp"], errors="coerce")
    telemetry = telemetry.dropna(subset=["timestamp"]).set_index("timestamp")
    if "total_latency_ms" in telemetry:
        _save_series_plot(
            telemetry["total_latency_ms"],
            "line",
            out_dir / "requests_latency_over_time.png",
            ylabel="ms",
        )
    if "estimated_cost_usd" in telemetry and telemetry["estimated_cost_usd"].notna().any():
        _save_series_plot(
            telemetry["estimated_cost_usd"].fillna(0).cumsum(),
            "line",
            out_dir / "estimated_cost_trend.png",
            ylabel="USD",
        )
    if "no_answer_numeric" in telemetry:
        _save_series_plot(
            telemetry["no_answer_numeric"].rolling(20, min_periods=1).mean(),
            "line",
            out_dir / "no_answer_rate_trend.png",
            ylim=(0, 1),
        )
    if "top_retrieval_score" in telemetry:
        _save_series_plot(
            telemetry["top_retrieval_score"].rolling(20, min_periods=1).mean(),
            "line",
            out_dir / "retrieval_score_trend.png",
        )

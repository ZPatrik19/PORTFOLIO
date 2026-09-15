"""Command-line entry point for indexing, benchmarking and local operations."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

import pandas as pd

from .config import load_config, resolve_path
from .evaluation import benchmark_retrieval_methods, generate_eval_dataset, run_retrieval_benchmark
from .figures import save_basic_figures
from .multi_index import MultiIndexManager
from .orchestration import KnowledgePlatform
from .public_docs import download_public_docs

DEFAULT_INDEX_VARIANTS = ["fixed", "recursive", "structure_aware", "semantic"]
DEFAULT_BENCHMARK_SAMPLES = 300


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without executing application logic."""

    parser = argparse.ArgumentParser(description="Technical Knowledge Intelligence CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ["download-public", "ingest", "index", "benchmark", "figures", "list-indexes"]:
        subparsers.add_parser(name)

    variants_parser = subparsers.add_parser("index-variants")
    variants_parser.add_argument("--strategies", nargs="+", default=DEFAULT_INDEX_VARIANTS)
    variants_parser.add_argument("--chunk-size", type=int)
    variants_parser.add_argument("--overlap", type=int)
    variants_parser.add_argument("--force", action="store_true")

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--index-variant", default="primary")
    ask_parser.add_argument("--prompt-optimization", default="none")
    return parser


def _run_benchmark(platform: KnowledgePlatform, config: dict) -> None:
    samples = generate_eval_dataset(platform.chunks, DEFAULT_BENCHMARK_SAMPLES)
    evaluation_dir = resolve_path(config["paths"]["evaluation"])
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    (evaluation_dir / "evaluation_300.json").write_text(
        json.dumps(samples, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    benchmark = run_retrieval_benchmark(platform.retriever, samples)
    raw_methods, summary = benchmark_retrieval_methods(platform.retriever, samples, config)
    benchmark_dir = resolve_path(config["paths"]["results"]) / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    benchmark.to_csv(benchmark_dir / "retrieval_benchmark.csv", index=False)
    raw_methods.to_csv(benchmark_dir / "retrieval_methods_raw.csv", index=False)
    summary.to_csv(benchmark_dir / "retrieval_methods_summary.csv", index=False)
    print(summary.to_json(orient="records", indent=2))


def _run_figures(platform: KnowledgePlatform, config: dict) -> None:
    processed_dir = resolve_path(config["paths"]["processed"])
    quality_path = processed_dir / "quality_chunks.csv"
    quality = pd.read_csv(quality_path) if quality_path.exists() else pd.DataFrame()

    benchmark_dir = resolve_path(config["paths"]["results"]) / "benchmarks"
    benchmark_path = benchmark_dir / "retrieval_benchmark.csv"
    method_summary_path = benchmark_dir / "retrieval_methods_summary.csv"
    benchmark = pd.read_csv(benchmark_path) if benchmark_path.exists() else pd.DataFrame()
    method_summary = (
        pd.read_csv(method_summary_path)
        if method_summary_path.exists()
        else pd.DataFrame()
    )
    telemetry = pd.DataFrame(platform.telemetry.recent(1000))
    save_basic_figures(
        platform.chunks,
        quality,
        benchmark,
        resolve_path(config["paths"]["results"]) / "figures",
        method_summary,
        telemetry,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one CLI command and return a process-compatible status code."""

    arguments = build_parser().parse_args(argv)
    config = load_config()

    if arguments.command == "download-public":
        report = download_public_docs(resolve_path(config["paths"]["reference_docs"]))
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    if arguments.command in {"ingest", "index"}:
        report = KnowledgePlatform(config).ingest_and_index()
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    if arguments.command == "ask":
        platform = KnowledgePlatform(config).load_index()
        answer = platform.ask(
            {
                "question": arguments.question,
                "debug": True,
                "index_variant": arguments.index_variant,
                "prompt_optimization": arguments.prompt_optimization,
            }
        )
        print(answer.model_dump_json(indent=2))
        return 0

    if arguments.command == "index-variants":
        result = MultiIndexManager(config).build(
            arguments.strategies,
            chunk_size=arguments.chunk_size,
            overlap=arguments.overlap,
            force=arguments.force,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if arguments.command == "list-indexes":
        print(json.dumps(MultiIndexManager(config).available(), indent=2, ensure_ascii=False))
        return 0

    platform = KnowledgePlatform(config).load_index()
    if arguments.command == "benchmark":
        _run_benchmark(platform, config)
        return 0
    if arguments.command == "figures":
        _run_figures(platform, config)
        return 0

    raise RuntimeError(f"Unhandled CLI command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())

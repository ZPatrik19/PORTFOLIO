from __future__ import annotations

import pandas as pd

from tkip.config import load_config, resolve_path


def classify_failure(row: pd.Series) -> str:
    """Classify deterministic retrieval failures from benchmark metrics."""
    if float(row.get("hit_rate", 0.0)) == 0.0:
        return "RETRIEVAL_FAILURE"
    if float(row.get("recall@5", 0.0)) < 1.0:
        return "RERANKING_FAILURE"
    return "PASS"


def main() -> int:
    """Create a simple failure-analysis CSV from the latest retrieval benchmark."""
    config = load_config()
    results_dir = resolve_path(config["paths"]["results"])
    benchmark_path = results_dir / "benchmarks" / "retrieval_benchmark.csv"
    output_path = results_dir / "evaluation" / "failure_report.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not benchmark_path.exists():
        print("No benchmark file found. Run the benchmark command first.")
        return 0

    benchmark = pd.read_csv(benchmark_path)
    benchmark["failure_type"] = benchmark.apply(classify_failure, axis=1)
    benchmark.to_csv(output_path, index=False)
    print(benchmark["failure_type"].value_counts().to_string())
    print(f"Failure report written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


import pandas as pd

from prompt_benchmark.paths import PATHS
from prompt_benchmark.utils.logging import configure_logging
from prompt_benchmark.benchmark.runner import run_strategy
from prompt_benchmark.evaluation.metrics import classification_metrics
from prompt_benchmark.data.benchmark_suites import representative_stratified_subset
from prompt_benchmark.llm.client import MockLLMClient
from prompt_benchmark.prompts import get_strategy, list_strategies


def main() -> None:
    """Run a fast offline end-to-end smoke check without mutating portfolio results."""
    configure_logging()
    benchmark_path = PATHS.processed_data / "benchmark.csv"
    if not benchmark_path.exists():
        subprocess.run([sys.executable, "05_scripts/02_prepare_data.py", "--source", "mock"], cwd=PATHS.root, check=True)

    benchmark = representative_stratified_subset(pd.read_csv(benchmark_path), 12, random_seed=42)
    client = MockLLMClient()

    with tempfile.TemporaryDirectory(prefix="prompt_bench_smoke_") as tmp:
        tmp_path = Path(tmp)
        for name in list_strategies():
            result = run_strategy(
                benchmark,
                get_strategy(name),
                client,
                tmp_path / f"{name}.csv",
                0.0,
                0.0,
                force=True,
            )
            metrics = classification_metrics(result)
            if metrics["error_rate"] != 0.0:
                raise RuntimeError(f"Smoke test failed for {name}: provider error detected")

    subprocess.run([sys.executable, "05_scripts/08_generate_diagrams.py"], cwd=PATHS.root, check=True)
    print("Smoke test completed successfully without modifying 07_outputs/results/. Mock scores are not LLM benchmark evidence.")


if __name__ == "__main__":
    main()

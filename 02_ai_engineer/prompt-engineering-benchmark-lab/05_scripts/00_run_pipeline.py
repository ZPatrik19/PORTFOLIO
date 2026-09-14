"""Run the end-to-end benchmark workflow in the documented order."""
from __future__ import annotations

import argparse
import subprocess
import sys

from prompt_benchmark.paths import PATHS
from prompt_benchmark.prompts import list_strategies
from prompt_benchmark.utils.logging import configure_logging

PROFILE_LIMITS = {"smoke": 12, "pilot": 36, "standard": 120, "strong": 300, "full": None}


def run(command: list[str]) -> None:
    subprocess.run([sys.executable, *command], cwd=PATHS.root, check=True)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Prepare data, benchmark prompts and generate reports.")
    parser.add_argument("--provider", default="mock")
    parser.add_argument("--source", choices=["mock", "huggingface", "sample"], default="mock")
    parser.add_argument("--profile", choices=list(PROFILE_LIMITS), default="standard")
    parser.add_argument("--strategy", choices=["all", *list_strategies()], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run(["05_scripts/02_prepare_data.py", "--source", args.source])
    benchmark_cmd = [
        "05_scripts/04_run_benchmark.py",
        "--provider",
        args.provider,
        "--strategy",
        args.strategy,
    ]
    limit = PROFILE_LIMITS[args.profile]
    if limit is not None:
        benchmark_cmd += ["--limit", str(limit)]
    if args.force:
        benchmark_cmd.append("--force")
    run(benchmark_cmd)
    run(["05_scripts/07_generate_report.py", "--provider", args.provider])


if __name__ == "__main__":
    main()

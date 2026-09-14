from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    command = [sys.executable, *args]
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a small real-LLM benchmark intended to fit comfortably inside common free-tier quotas."
    )
    parser.add_argument("--provider", choices=["ollama", "groq", "gemini"], default="groq")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--with-ablation", action="store_true")
    args = parser.parse_args()

    if not (PROJECT_ROOT / "01_data/processed/benchmark.csv").exists():
        run("05_scripts/02_prepare_data.py", "--source", "mock")

    run("05_scripts/03_check_provider.py", "--provider", args.provider)
    run(
        "05_scripts/04_run_benchmark.py",
        "--provider",
        args.provider,
        "--strategy",
        "all",
        "--limit",
        str(args.limit),
    )
    run("05_scripts/07_generate_report.py", "--provider", args.provider)
    if args.with_ablation:
        run(
            "05_scripts/05_run_ablation.py",
            "--provider",
            args.provider,
            "--limit",
            str(args.limit),
        )

    print(
        "\nFree demo completed. For Groq/Gemini, free-tier quotas can change; "
        "the CSV checkpoints allow later continuation."
    )


if __name__ == "__main__":
    main()

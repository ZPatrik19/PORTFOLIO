from __future__ import annotations

from pathlib import Path


import argparse

from prompt_benchmark.paths import PATHS
from prompt_benchmark.utils.logging import configure_logging
from prompt_benchmark.data.mock_generator import save_mock_support_tickets


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Generate local synthetic support-ticket data.")
    parser.add_argument("--samples-per-class", type=int, default=120)
    parser.add_argument("--output", default=str(PATHS.mock_data / "mock_support_tickets.csv"))
    args = parser.parse_args()
    path = save_mock_support_tickets(args.output, args.samples_per_class)
    print(f"Saved {args.samples_per_class * 6} synthetic rows -> {path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path


from prompt_benchmark.paths import PATHS
from prompt_benchmark.utils.logging import configure_logging
from prompt_benchmark.evaluation.diagrams import generate_all


if __name__ == "__main__":
    generate_all()
    print("Saved conceptual diagrams to 07_outputs/reports/figures/.")

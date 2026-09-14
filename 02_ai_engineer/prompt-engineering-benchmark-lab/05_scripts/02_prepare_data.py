from __future__ import annotations

from pathlib import Path

import argparse
import logging

from prompt_benchmark.paths import PATHS
from prompt_benchmark.utils.logging import configure_logging
from prompt_benchmark.config import load_yaml
from prompt_benchmark.data.mock_generator import save_mock_support_tickets
from prompt_benchmark.constants import LABELS
LOGGER = logging.getLogger(__name__)

from prompt_benchmark.data.prepare_benchmark import (
    load_huggingface_dataframe,
    load_sample_dataframe,
    prepare_splits,
    save_prepared_data,
)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Prepare disjoint development/benchmark/few-shot data.")
    parser.add_argument(
        "--source",
        choices=["mock", "huggingface", "sample"],
        default="mock",
        help=("mock is a fully local synthetic challenge suite; huggingface creates a controlled "
              "split from the public synthetic support-ticket dataset; use the external HF suite "
              "for official train/test generalization checks."),
    )
    parser.add_argument("--benchmark-per-class", type=int, default=None)
    parser.add_argument("--development-per-class", type=int, default=None)
    parser.add_argument("--few-shot-per-class", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(PATHS.configs / "benchmark.yaml")
    benchmark_per_class = (
        args.benchmark_per_class
        if args.benchmark_per_class is not None
        else int(cfg["benchmark_samples_per_class"])
    )
    development_per_class = (
        args.development_per_class
        if args.development_per_class is not None
        else int(cfg["development_samples_per_class"])
    )
    few_shot_per_class = (
        args.few_shot_per_class
        if args.few_shot_per_class is not None
        else int(cfg["few_shot_examples_per_class"])
    )

    if args.source == "huggingface":
        frame = load_huggingface_dataframe()
    elif args.source == "sample":
        frame = load_sample_dataframe("01_data/sample/support_tickets_sample.csv")
    else:
        mock_path = PATHS.mock_data / "mock_support_tickets.csv"
        required_per_class = benchmark_per_class + development_per_class + few_shot_per_class
        configured_source = int(cfg.get("mock_source_samples_per_class", 1800))
        target_per_class = max(configured_source, required_per_class + 200)
        regenerate = not mock_path.exists()
        if not regenerate:
            try:
                existing = load_sample_dataframe(mock_path)
                counts = existing["label"].astype(str).str.lower().value_counts()
                required_cases = {"primary_last", "conditional_distractor", "code_log_noise", "label_word_attack", "double_negation"}
                present_cases = set(existing.get("case_type", []).astype(str)) if "case_type" in existing.columns else set()
                regenerate = counts.reindex(LABELS, fill_value=0).min() < required_per_class or not required_cases.issubset(present_cases)
            except (OSError, ValueError, KeyError) as exc:
                LOGGER.warning("Existing mock dataset is incompatible and will be regenerated: %s", exc)
                regenerate = True
        if regenerate:
            save_mock_support_tickets(mock_path, samples_per_class=target_per_class)
        frame = load_sample_dataframe(mock_path)

    benchmark, development, few_shot = prepare_splits(
        frame,
        benchmark_per_class,
        development_per_class,
        few_shot_per_class,
        int(cfg["random_seed"]),
    )
    save_prepared_data(
        benchmark,
        development,
        few_shot,
        PATHS.processed_data,
        PATHS.prompt_examples / "few_shot_examples.json",
    )
    print(
        f"Source={args.source} | Saved benchmark={len(benchmark)}, "
        f"development={len(development)}, few-shot={len(few_shot)}"
    )
    if args.source == "mock":
        print("NOTE: Synthetic mock data is for runnable demos. For portfolio evidence, use a real LLM provider and an external or human-reviewed evaluation set; the HF suite is an additional synthetic generalization check.")


if __name__ == "__main__":
    main()

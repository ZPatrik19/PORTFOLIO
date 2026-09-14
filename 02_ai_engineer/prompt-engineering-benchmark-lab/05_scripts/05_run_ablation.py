from __future__ import annotations

from pathlib import Path


import argparse
import os

import pandas as pd
from dotenv import load_dotenv

from prompt_benchmark.paths import PATHS
from prompt_benchmark.utils.logging import configure_logging
from prompt_benchmark.benchmark.runner import run_strategy
from prompt_benchmark.config import load_yaml
from prompt_benchmark.evaluation.metrics import classification_metrics
from prompt_benchmark.evaluation.plots import bar_metric
from prompt_benchmark.llm.factory import SUPPORTED_PROVIDERS, create_llm_client
from prompt_benchmark.prompts.base import PromptPayload
from prompt_benchmark.prompts.strategies import CONSTRAINTS, DECISION_POLICY, DEFINITIONS, ROLE, _few_shot_text
from prompt_benchmark.utils.pricing import get_provider_pricing


class AblationStrategy:
    def __init__(
        self,
        name: str,
        include_definitions: bool,
        include_few_shot: bool,
        include_constraints: bool,
        include_policy: bool,
    ) -> None:
        self.name = name
        self.flags = (include_definitions, include_few_shot, include_constraints, include_policy)

    def build(self, ticket: str) -> PromptPayload:
        definitions, few_shot, constraints, policy = self.flags
        parts: list[str] = []
        if definitions:
            parts.append(DEFINITIONS)
        if few_shot:
            parts.append(_few_shot_text(PATHS.prompt_examples / "few_shot_examples.json"))
        if constraints:
            parts.append(CONSTRAINTS)
        if policy:
            parts.append(DECISION_POLICY)
        parts.append(f"Message: {ticket}\nReturn exactly one allowed label.")
        return PromptPayload(self.name, ROLE, "\n\n".join(parts))


def main() -> None:
    configure_logging()
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run prompt-component ablations on the development split.")
    parser.add_argument(
        "--provider",
        choices=list(SUPPORTED_PROVIDERS),
        default=os.getenv("LLM_PROVIDER", "mock"),
    )
    parser.add_argument("--dataset", default=str(PATHS.processed_data / "development.csv"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = PATHS.root / dataset_path
    data = pd.read_csv(dataset_path)
    cfg = load_yaml(PATHS.configs / "benchmark.yaml")
    pricing = load_yaml(PATHS.configs / "pricing.yaml")
    input_price, output_price = get_provider_pricing(pricing, args.provider)
    client = create_llm_client(args.provider, cfg)

    variants = [
        AblationStrategy("full", True, True, True, True),
        AblationStrategy("minus_definitions", False, True, True, True),
        AblationStrategy("minus_few_shot", True, False, True, True),
        AblationStrategy("minus_constraints", True, True, False, True),
        AblationStrategy("minus_decision_policy", True, True, True, False),
    ]

    rows: list[dict[str, object]] = []
    for strategy in variants:
        result = run_strategy(
            data,
            strategy,
            client,
            PATHS.results / "raw" / args.provider / f"ablation_{strategy.name}.csv",
            input_price,
            output_price,
            args.limit,
            args.force,
        )
        rows.append(
            {
                "provider": args.provider,
                "model": client.model,
                "strategy": strategy.name,
                **classification_metrics(result),
            }
        )

    summary = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    output_dir = PATHS.results / args.provider
    figure_dir = PATHS.figures / args.provider
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "ablation_results.csv", index=False)
    bar_metric(
        summary,
        "macro_f1",
        f"Prompt Ablation: Macro F1 ({args.provider})",
        "Macro F1",
        figure_dir / "ablation_macro_f1.png",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

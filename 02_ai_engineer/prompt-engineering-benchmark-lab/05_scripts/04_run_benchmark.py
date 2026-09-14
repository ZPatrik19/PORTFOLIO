from __future__ import annotations

import argparse
import os
from pathlib import Path


import pandas as pd
from dotenv import load_dotenv

from prompt_benchmark.paths import PATHS
from prompt_benchmark.utils.logging import configure_logging
from prompt_benchmark.benchmark.runner import run_strategy
from prompt_benchmark.config import load_yaml
from prompt_benchmark.llm.factory import SUPPORTED_PROVIDERS, create_llm_client
from prompt_benchmark.prompts import get_strategy, list_strategies
from prompt_benchmark.utils.pricing import get_provider_pricing


def main() -> None:
    configure_logging()
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the prompt-engineering benchmark.")
    parser.add_argument("--provider", choices=list(SUPPORTED_PROVIDERS), default=os.getenv("LLM_PROVIDER", "mock"))
    parser.add_argument("--strategy", default="all", choices=["all", *list_strategies()])
    parser.add_argument("--dataset", default=str(PATHS.processed_data / "benchmark.csv"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    benchmark_path = Path(args.dataset)
    if not benchmark_path.is_absolute():
        benchmark_path = PATHS.root / benchmark_path
    if not benchmark_path.exists():
        raise FileNotFoundError(f"{benchmark_path} does not exist. Run 05_scripts/02_prepare_data.py first.")

    benchmark = pd.read_csv(benchmark_path)
    cfg = load_yaml(PATHS.configs / "benchmark.yaml")
    pricing = load_yaml(PATHS.configs / "pricing.yaml")
    names = list_strategies() if args.strategy == "all" else [args.strategy]

    if args.dry_run:
        sample = str(benchmark.iloc[0]["text"])
        for name in names:
            strategy = get_strategy(name)
            if hasattr(strategy, "build_branches"):
                payloads = strategy.build_branches(sample)
                print(f"\n--- {name}: {len(payloads)} branches ---")
                for idx, payload in enumerate(payloads, 1):
                    print(f"\n[branch {idx}]\nSystem: {payload.instructions}\n{payload.input_text}")
            else:
                payload = strategy.build(sample)
                print(f"\n--- {name} ---\nSystem: {payload.instructions}\n{payload.input_text}\nstructured={payload.structured_output}\nreasoning={payload.reasoning_effort}")
        return

    overrides = {k: v for k, v in {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "model": args.model,
    }.items() if v is not None}
    client = create_llm_client(args.provider, cfg, overrides=overrides)
    input_price, output_price = get_provider_pricing(pricing, args.provider)
    print(f"Provider={client.provider} Model={client.model} Settings={client.generation_settings()}")

    for name in names:
        strategy = get_strategy(name)
        path = PATHS.results / "raw" / args.provider / f"{name}.csv"
        result = run_strategy(
            benchmark=benchmark,
            strategy=strategy,
            client=client,
            output_path=path,
            input_price_per_million=input_price,
            output_price_per_million=output_price,
            limit=args.limit,
            force=args.force,
        )
        print(f"{args.provider}:{name}: {len(result)} rows -> {path}")


if __name__ == "__main__":
    main()

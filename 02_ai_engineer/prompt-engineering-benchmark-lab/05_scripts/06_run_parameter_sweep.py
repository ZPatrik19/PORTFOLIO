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
from prompt_benchmark.evaluation.metrics import classification_metrics
from prompt_benchmark.llm.factory import PROVIDER_CAPABILITIES, SUPPORTED_PROVIDERS, create_llm_client
from prompt_benchmark.prompts import get_strategy, list_strategies
from prompt_benchmark.utils.pricing import get_provider_pricing


def _slug(name: str, value: float | int) -> str:
    return f"{name}_{str(value).replace('.', 'p')}"


def main() -> None:
    configure_logging()
    load_dotenv()
    parser = argparse.ArgumentParser(description="Benchmark temperature/top_p/top_k separately.")
    parser.add_argument("--provider", choices=list(SUPPORTED_PROVIDERS), default=os.getenv("LLM_PROVIDER", "mock"))
    parser.add_argument("--strategy", choices=list_strategies(), default=None)
    parser.add_argument("--dataset", default=str(PATHS.processed_data / "development.csv"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(PATHS.configs / "benchmark.yaml")
    sweep_cfg = load_yaml(PATHS.configs / "parameter_sweeps.yaml")
    pricing = load_yaml(PATHS.configs / "pricing.yaml")
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = PATHS.root / dataset_path
    data = pd.read_csv(dataset_path)
    strategy_name = args.strategy or str(sweep_cfg["strategy"])
    strategy = get_strategy(strategy_name)
    limit = args.limit if args.limit is not None else int(sweep_cfg.get("limit", 60))
    input_price, output_price = get_provider_pricing(pricing, args.provider)
    caps = PROVIDER_CAPABILITIES[args.provider]

    experiments: list[tuple[str, float | int, dict[str, float | int | None]]] = []
    if caps["temperature"]:
        for value in sweep_cfg["temperature_values"]:
            experiments.append(("temperature", value, {"temperature": float(value), "top_p": None, "top_k": None}))
    if caps["top_p"]:
        for value in sweep_cfg["top_p_values"]:
            # Hold temperature low/fixed so top_p is the manipulated variable.
            experiments.append(("top_p", value, {"temperature": 0.0, "top_p": float(value), "top_k": None}))
    if caps["top_k"]:
        for value in sweep_cfg["top_k_values"]:
            experiments.append(("top_k", value, {"temperature": 0.0, "top_p": None, "top_k": int(value)}))

    rows: list[dict[str, object]] = []
    raw_root = PATHS.results / "parameter_sweeps" / args.provider
    raw_root.mkdir(parents=True, exist_ok=True)

    for parameter, value, overrides in experiments:
        client = create_llm_client(args.provider, cfg, overrides=overrides)
        path = raw_root / f"{strategy_name}__{_slug(parameter, value)}.csv"
        result = run_strategy(data, strategy, client, path, input_price, output_price, limit=limit, force=args.force)
        rows.append({
            "provider": args.provider,
            "model": client.model,
            "strategy": strategy_name,
            "parameter": parameter,
            "value": value,
            "temperature": client.temperature,
            "top_p": client.top_p,
            "top_k": client.top_k,
            **classification_metrics(result),
        })

    summary = pd.DataFrame(rows)
    out_dir = PATHS.results / "parameter_sweeps" / args.provider
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "parameter_sweep_summary.csv", index=False)

    # Save portfolio-ready figures with one chart per decoding parameter.
    import matplotlib.pyplot as plt
    figure_dir = PATHS.figures / args.provider / "parameter_sweeps"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for parameter, group in summary.groupby("parameter"):
        ordered = group.sort_values("value")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ordered["value"], ordered["macro_f1"], marker="o")
        ax.set_title(f"{parameter} vs Macro F1 — {args.provider}")
        ax.set_xlabel(parameter)
        ax.set_ylabel("Macro F1")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{parameter}_vs_macro_f1.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    print(summary[["parameter", "value", "macro_f1", "invalid_output_rate", "mean_total_tokens", "p95_latency_seconds"]].to_string(index=False))
    print(f"\nSaved -> {out_dir / 'parameter_sweep_summary.csv'}")
    print(f"Figures -> {figure_dir}")


if __name__ == "__main__":
    main()

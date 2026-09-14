from __future__ import annotations

import argparse
import os
from dotenv import load_dotenv

from prompt_benchmark.paths import PATHS
from prompt_benchmark.utils.logging import configure_logging
from prompt_benchmark.config import load_yaml
from prompt_benchmark.evaluation.parsing import parse_prediction
from prompt_benchmark.llm.factory import SUPPORTED_PROVIDERS, create_llm_client
from prompt_benchmark.prompts import list_custom_prompts, load_custom_prompt


def main() -> None:
    configure_logging()
    load_dotenv()
    presets = {p.stem: p for p in list_custom_prompts()}
    parser = argparse.ArgumentParser(description="Run a saved custom prompt preset with any supported provider.")
    parser.add_argument("--provider", choices=list(SUPPORTED_PROVIDERS), default=os.getenv("LLM_PROVIDER", "mock"))
    parser.add_argument("--preset", required=True, choices=sorted(presets) if presets else None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--ticket", default="Please cancel my subscription before the next renewal.")
    args = parser.parse_args()
    if args.preset not in presets:
        raise SystemExit(f"Unknown preset: {args.preset}. Save one in the UI first.")

    cfg = load_yaml(PATHS.configs / "benchmark.yaml")
    overrides = {"model": args.model} if args.model else None
    client = create_llm_client(args.provider, cfg, overrides=overrides)
    strategy = load_custom_prompt(presets[args.preset])
    payload = strategy.build(args.ticket)
    response = client.classify(payload)
    parsed = parse_prediction(response.raw_output, payload.output_mode)

    print(f"provider      : {response.provider}")
    print(f"model         : {response.model}")
    print(f"preset        : {args.preset}")
    print(f"prediction    : {parsed.label}")
    print(f"valid_output  : {parsed.valid_output}")
    print(f"valid_json    : {parsed.valid_json}")
    print(f"input_tokens  : {response.input_tokens}")
    print(f"output_tokens : {response.output_tokens}")
    print(f"total_tokens  : {response.total_tokens}")
    print(f"latency       : {response.latency_seconds:.4f}s")
    print(f"error         : {response.error}")
    print(f"raw_output    : {response.raw_output}")


if __name__ == "__main__":
    main()

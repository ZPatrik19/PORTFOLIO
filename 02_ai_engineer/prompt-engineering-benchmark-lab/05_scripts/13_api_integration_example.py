from __future__ import annotations

import argparse
import os
from dotenv import load_dotenv

from prompt_benchmark.paths import PATHS
from prompt_benchmark.utils.logging import configure_logging
from prompt_benchmark.config import load_yaml
from prompt_benchmark.llm.factory import SUPPORTED_PROVIDERS, create_llm_client
from prompt_benchmark.prompts import get_strategy, list_strategies


def main() -> None:
    configure_logging()
    load_dotenv()
    parser = argparse.ArgumentParser(description="Send one ticket through any configured provider and print telemetry.")
    parser.add_argument("--provider", choices=list(SUPPORTED_PROVIDERS), default=os.getenv("LLM_PROVIDER", "mock"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--strategy", choices=list_strategies(), default="p16_full_advanced_template")
    parser.add_argument(
        "--ticket",
        default="The old invoice issue is resolved. My current request is to cancel before the next renewal.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(PATHS.configs / "benchmark.yaml")
    overrides = {
        key: value
        for key, value in {
            "model": args.model,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        }.items()
        if value is not None
    }
    client = create_llm_client(args.provider, cfg, overrides=overrides)
    payload = get_strategy(args.strategy).build(args.ticket)
    response = client.classify(payload)

    print(f"provider       : {response.provider}")
    print(f"model          : {response.model}")
    print(f"strategy       : {args.strategy}")
    print(f"settings       : {client.generation_settings()}")
    print(f"input_tokens   : {response.input_tokens} ({response.token_source})")
    print(f"output_tokens  : {response.output_tokens} ({response.token_source})")
    print(f"total_tokens   : {response.total_tokens}")
    print(f"latency_seconds: {response.latency_seconds:.4f} ({response.latency_source})")
    print(f"error          : {response.error}")
    print(f"raw_output     : {response.raw_output}")


if __name__ == "__main__":
    main()

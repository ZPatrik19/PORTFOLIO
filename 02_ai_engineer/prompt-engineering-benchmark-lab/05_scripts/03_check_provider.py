from __future__ import annotations

from pathlib import Path


import argparse
import os

from dotenv import load_dotenv

from prompt_benchmark.paths import PATHS
from prompt_benchmark.utils.logging import configure_logging
from prompt_benchmark.config import load_yaml
from prompt_benchmark.llm.factory import SUPPORTED_PROVIDERS, create_llm_client
from prompt_benchmark.prompts import get_strategy
from prompt_benchmark.evaluation.parsing import parse_prediction


def main() -> None:
    configure_logging()
    load_dotenv()
    parser = argparse.ArgumentParser(description="Make exactly one provider request to verify configuration.")
    parser.add_argument(
        "--provider",
        choices=list(SUPPORTED_PROVIDERS),
        default=os.getenv("LLM_PROVIDER", "mock"),
    )
    parser.add_argument("--structured", action="store_true", help="Test P7 structured output instead of P0.")
    args = parser.parse_args()

    cfg = load_yaml(PATHS.configs / "benchmark.yaml")
    client = create_llm_client(args.provider, cfg)
    strategy = get_strategy("p7_structured_output" if args.structured else "p0_zero_shot")
    payload = strategy.build("I was charged twice for the same monthly subscription.")
    response = client.classify(payload)
    parsed = parse_prediction(response.raw_output, payload.output_mode)

    print(f"Provider : {response.provider}")
    print(f"Model    : {response.model}")
    print(f"Raw      : {response.raw_output!r}")
    print(f"Parsed   : {parsed.label}")
    print(f"Valid    : {parsed.valid_output}")
    print(f"Tokens   : {response.input_tokens} in / {response.output_tokens} out")
    print(f"Latency  : {response.latency_seconds:.3f} s")
    print(f"Error    : {response.error or 'None'}")
    if response.error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

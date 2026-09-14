from __future__ import annotations

import argparse
from dotenv import load_dotenv

from prompt_benchmark.paths import PATHS
from prompt_benchmark.utils.logging import configure_logging
from prompt_benchmark.llm.gemini_health import test_gemini_connection


def main() -> None:
    configure_logging()
    load_dotenv(PATHS.root / ".env")
    parser = argparse.ArgumentParser(description="Safely test a Google AI Studio / Gemini API key without printing it.")
    parser.add_argument("--model", default=None, help="Override GEMINI_MODEL for this test.")
    parser.add_argument("--structured", action="store_true", help="Also request JSON structured output.")
    args = parser.parse_args()

    result = test_gemini_connection(model=args.model, structured=args.structured)
    print(f"Gemini connection : {'OK' if result.ok else 'FAILED'}")
    print(f"Model             : {result.model}")
    print(f"HTTP status       : {result.http_status if result.http_status is not None else '-'}")
    print(f"Response          : {result.text or '-'}")
    print(f"Input tokens      : {result.input_tokens}")
    print(f"Output tokens     : {result.output_tokens}")
    print(f"Total tokens      : {result.total_tokens}")
    if result.error:
        print(f"Error             : {result.error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

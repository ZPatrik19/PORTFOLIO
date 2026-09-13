from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace

from travel_agent.agent import build_agent, resolved_methodology
from travel_agent.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic Travel Research Assistant")
    parser.add_argument("query", nargs="*", help="Travel question. If omitted, interactive mode starts.")
    parser.add_argument("--trace", action="store_true", help="Print tool trace after the answer.")
    parser.add_argument("--language", choices=["en", "hu"], help="Override APP_LANGUAGE for this run.")
    parser.add_argument(
        "--methodology",
        choices=["rule_based", "plan_execute", "ml_router", "openai_direct"],
        help="Override AGENT_METHODOLOGY for this run.",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.language:
        settings = replace(settings, language=args.language)
    if args.methodology:
        settings = replace(settings, methodology=args.methodology)
    agent = build_agent(settings)
    methodology = resolved_methodology(settings)

    if args.query:
        queries = [" ".join(args.query)]
    else:
        print(f"Mode: {settings.mode} | methodology: {methodology} | language: {settings.language}. Type 'quit' to exit.")
        while True:
            query = input("\nYou> ").strip()
            if query.lower() in {"quit", "exit"}:
                break
            if query:
                run = agent.run(query)
                print(f"\nAssistant> {run.answer}")
                if args.trace:
                    print(json.dumps([asdict(x) for x in run.trace], ensure_ascii=False, indent=2))
                    if run.metadata:
                        print("\nMETADATA")
                        print(json.dumps(run.metadata, ensure_ascii=False, indent=2))
        return

    for query in queries:
        run = agent.run(query)
        print(run.answer)
        if args.trace:
            print("\nTOOL TRACE")
            print(json.dumps([asdict(x) for x in run.trace], ensure_ascii=False, indent=2))
            if run.metadata:
                print("\nMETADATA")
                print(json.dumps(run.metadata, ensure_ascii=False, indent=2))
            print(f"\nTotal latency: {run.total_latency_ms:.2f} ms")


if __name__ == "__main__":
    main()

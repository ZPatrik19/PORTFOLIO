from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "03_src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from travel_agent.agent import MLRouterTravelAgent, PlanThenExecuteTravelAgent
from travel_agent.presets import PRESET_QUESTIONS


def build_agent(methodology: str, language: str):
    if methodology == "plan_execute":
        return PlanThenExecuteTravelAgent(language=language)
    if methodology == "ml_router":
        return MLRouterTravelAgent(language=language)
    raise ValueError(methodology)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate all bilingual UI preset scenarios.")
    parser.add_argument("--methodologies", nargs="+", choices=["plan_execute", "ml_router"], default=["plan_execute", "ml_router"])
    parser.add_argument("--output-dir", default=str(ROOT / "06_results" / "preset_validation"))
    args = parser.parse_args()

    os.environ["TRAVEL_DATA_MODE"] = "local"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for methodology in args.methodologies:
        for language in ("hu", "en"):
            agent = build_agent(methodology, language)
            for item in PRESET_QUESTIONS:
                run = agent.run(item[language])
                expected = item["expected_tools"]
                actual = run.tool_names
                rows.append({
                    "scenario_id": item["id"],
                    "language": language,
                    "methodology": methodology,
                    "category": item["category"],
                    "difficulty": item["difficulty"],
                    "expected_tools": "|".join(expected),
                    "actual_tools": "|".join(actual),
                    "route_exact": actual == expected,
                    "all_calls_success": all(call.success for call in run.trace),
                    "tool_calls": len(run.trace),
                    "latency_ms": round(run.total_latency_ms, 3),
                })

    with (output_dir / "preset_validation.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, dict] = {}
    for methodology in args.methodologies:
        subset = [row for row in rows if row["methodology"] == methodology]
        summary[methodology] = {
            "runs": len(subset),
            "exact_routes": sum(bool(row["route_exact"]) for row in subset),
            "exact_route_rate": sum(bool(row["route_exact"]) for row in subset) / len(subset),
            "successful_runs": sum(bool(row["all_calls_success"]) for row in subset),
            "successful_run_rate": sum(bool(row["all_calls_success"]) for row in subset) / len(subset),
            "mean_latency_ms": sum(float(row["latency_ms"]) for row in subset) / len(subset),
        }
    summary["catalogue"] = {"scenarios": len(PRESET_QUESTIONS), "languages": 2, "total_methodology_runs": len(rows)}
    (output_dir / "preset_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

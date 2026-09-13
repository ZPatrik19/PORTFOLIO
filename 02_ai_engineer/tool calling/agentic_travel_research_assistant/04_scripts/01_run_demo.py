"""Run a multi-tool travel research request directly from the repository."""
from __future__ import annotations
import argparse, json, sys
from dataclasses import asdict, replace
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1];SRC=PROJECT_ROOT/"03_src";sys.path.insert(0,str(SRC)) if str(SRC) not in sys.path else None
from travel_agent.agent import build_agent, resolved_methodology
from travel_agent.config import get_settings

DEFAULT_QUERY=("I am going to Vienna for 3 days. Check the weather, find hotels under 150 EUR per night, "
               "show museums and landmarks, give me public transport costs, and convert 500 EUR to HUF.")

def main():
    p=argparse.ArgumentParser();p.add_argument("--query",default=DEFAULT_QUERY);p.add_argument("--methodology",choices=["rule_based","plan_execute","ml_router","openai_direct"],default="plan_execute");p.add_argument("--language",choices=["en","hu"],default="en");args=p.parse_args()
    settings=replace(get_settings(),methodology=args.methodology,language=args.language);agent=build_agent(settings);run=agent.run(args.query)
    print("QUERY\n",args.query,sep="");print(f"\nMETHODOLOGY: {resolved_methodology(settings)}")
    print("\nFINAL ANSWER\n",run.answer,sep="");print("\nTOOL TRACE")
    for call in run.trace:print(json.dumps(asdict(call),ensure_ascii=False,indent=2))
    print(f"\nTOTAL LATENCY: {run.total_latency_ms:.2f} ms")
if __name__=="__main__":main()

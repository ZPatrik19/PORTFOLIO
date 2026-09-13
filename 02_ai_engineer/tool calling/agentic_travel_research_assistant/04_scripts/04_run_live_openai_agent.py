"""Run the real OpenAI Responses API agent and print every function call/output."""
from __future__ import annotations
import json,os,sys
from dataclasses import asdict,replace
from pathlib import Path
os.environ["TRAVEL_DATA_MODE"]="auto"
PROJECT_ROOT=Path(__file__).resolve().parents[1];SRC=PROJECT_ROOT/"03_src";sys.path.insert(0,str(SRC)) if str(SRC) not in sys.path else None
from travel_agent.agent import build_agent
from travel_agent.config import get_settings

QUERY="Plan 3 days in Vienna: weather, hotels under 150 EUR, museums, public transport, and convert 500 EUR to HUF."

def main():
    settings=replace(get_settings(),mode="openai",methodology="openai_direct")
    if not settings.openai_api_key:raise SystemExit("OPENAI_API_KEY is missing. Copy .env.example to .env and add the key.")
    run=build_agent(settings).run(QUERY)
    print(run.answer);print("\nFUNCTION CALL TRACE")
    print(json.dumps([asdict(x) for x in run.trace],ensure_ascii=False,indent=2))
if __name__=="__main__":main()

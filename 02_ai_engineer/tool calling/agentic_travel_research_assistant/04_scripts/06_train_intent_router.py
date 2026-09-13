"""Train the bilingual multi-label intent/tool router."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"03_src"))
from travel_agent.training import train_router

if __name__ == "__main__":
    metrics,_=train_router(write_outputs=True)
    print(json.dumps(metrics,indent=2))

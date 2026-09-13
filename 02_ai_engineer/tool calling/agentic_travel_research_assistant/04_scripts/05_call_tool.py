"""Invoke one registered tool directly with JSON arguments."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1];SRC=PROJECT_ROOT/"03_src";sys.path.insert(0,str(SRC)) if str(SRC) not in sys.path else None
from travel_agent.tools import ToolRegistry

def main():
    p=argparse.ArgumentParser(description="Direct ToolRegistry invocation")
    p.add_argument("tool",choices=ToolRegistry().names)
    p.add_argument("--arguments",required=True,help='JSON object, e.g. {"city":"Vienna"}')
    args=p.parse_args();payload=json.loads(args.arguments)
    output,latency,success,error=ToolRegistry().execute(args.tool,payload)
    print(json.dumps({"tool":args.tool,"arguments":payload,"success":success,"latency_ms":latency,"error":error,"output":output},ensure_ascii=False,indent=2))
    raise SystemExit(0 if success else 1)
if __name__=="__main__":main()

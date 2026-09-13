"""One-command preparation: quality audit -> router training -> quick benchmark."""
from __future__ import annotations
import argparse, json, os, sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"03_src"));os.environ.setdefault("TRAVEL_DATA_MODE","local")
from travel_agent.agent import MLRouterTravelAgent
from travel_agent.evaluation.runner import evaluate_agent
from travel_agent.quality import audit_all
from travel_agent.training import train_router

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--limit",type=int,default=500);p.add_argument("--regenerate-data",action="store_true");a=p.parse_args()
    if a.regenerate_data:
        import subprocess
        subprocess.run([sys.executable,str(ROOT/"00_setup/04_generate_data.py")],check=True)
        subprocess.run([sys.executable,str(ROOT/"00_setup/05_upgrade_data_quality.py")],check=True)
    report=audit_all(write_outputs=True)
    if not all(report["quality_gates"].values()):
        raise SystemExit("Data-quality gates failed. Inspect 06_results/data_quality/data_quality_report.json")
    metrics,_=train_router(write_outputs=True)
    agent=MLRouterTravelAgent(language="hu")
    with tempfile.TemporaryDirectory(prefix="travel-agent-prepare-") as tmpdir:
        _,summary=evaluate_agent(agent,ROOT/"01_data/benchmark/agent_tasks.json",Path(tmpdir),a.limit)
    print(json.dumps({"quality_gate_pass_rate":report["quality_gate_pass_rate"],"router_training":metrics,"quick_agent_evaluation":summary},ensure_ascii=False,indent=2))

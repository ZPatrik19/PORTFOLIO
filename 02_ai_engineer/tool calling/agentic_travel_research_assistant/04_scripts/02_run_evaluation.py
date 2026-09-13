"""Evaluate one orchestration strategy on the 300-case benchmark."""
from __future__ import annotations
import argparse,os,sys
from dataclasses import replace
from pathlib import Path
os.environ["TRAVEL_DATA_MODE"]="local"
PROJECT_ROOT=Path(__file__).resolve().parents[1];SRC=PROJECT_ROOT/"03_src";sys.path.insert(0,str(SRC)) if str(SRC) not in sys.path else None
from travel_agent.agent import build_agent
from travel_agent.config import get_settings
from travel_agent.evaluation.plotting import plot_case_diagnostics,plot_efficiency,plot_latency,plot_metrics
from travel_agent.evaluation.runner import evaluate_agent

def main():
    p=argparse.ArgumentParser();p.add_argument("--methodology",choices=["rule_based","plan_execute","ml_router","openai_direct"],default="plan_execute");p.add_argument("--language",choices=["en","hu"],default="en");p.add_argument("--limit",type=int,default=None);args=p.parse_args()
    settings=replace(get_settings(),methodology=args.methodology,language=args.language);agent=build_agent(settings)
    dataset=PROJECT_ROOT/"01_data"/"benchmark"/"agent_tasks.json";output=PROJECT_ROOT/"06_results"/args.methodology
    rows,summary=evaluate_agent(agent,dataset,output,limit=args.limit)
    plot_metrics(output/"metrics.json",output/"quality_metrics.png");plot_efficiency(output/"metrics.json",output/"efficiency_metrics.png");plot_latency(output/"metrics.json",output/"latency_metrics.png");plot_case_diagnostics(output/"case_metrics.csv",output/"case_diagnostics.png")
    print(f"Evaluated {len(rows)} cases -> {output}")
    for k,v in summary.items():print(f"{k}: {v}")
if __name__=="__main__":main()

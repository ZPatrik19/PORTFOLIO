"""Compare offline orchestration strategies on the same benchmark."""
from __future__ import annotations
import argparse,csv,json,os,sys
from pathlib import Path
os.environ["TRAVEL_DATA_MODE"]="local"
PROJECT_ROOT=Path(__file__).resolve().parents[1];SRC=PROJECT_ROOT/"03_src";sys.path.insert(0,str(SRC)) if str(SRC) not in sys.path else None
from travel_agent.agent import OfflineTravelAgent,PlanThenExecuteTravelAgent,MLRouterTravelAgent
from travel_agent.evaluation.plotting import plot_methodology_comparison
from travel_agent.evaluation.runner import evaluate_agent

def main():
    p=argparse.ArgumentParser();p.add_argument("--limit",type=int,default=500,help="Cases per methodology; use 22500 for the full benchmark")
    args=p.parse_args()
    dataset=PROJECT_ROOT/"01_data"/"benchmark"/"agent_tasks.json";results=PROJECT_ROOT/"06_results";results.mkdir(parents=True,exist_ok=True)
    agents={"rule_based":OfflineTravelAgent(),"plan_execute":PlanThenExecuteTravelAgent(),"ml_router":MLRouterTravelAgent()};summaries={}
    for name,agent in agents.items():_,summaries[name]=evaluate_agent(agent,dataset,results/name,limit=args.limit)
    payload={"benchmark_limit":args.limit,"summaries":summaries}
    path=results/"methodology_comparison.json";path.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    metrics=sorted(next(iter(summaries.values())).keys())
    with (results/"methodology_comparison.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f);w.writerow(["methodology",*metrics]);[w.writerow([n,*[s[k] for k in metrics]]) for n,s in summaries.items()]
    # plotting helper expects the previous flat JSON shape, so write a temporary compatible file.
    compat=results/"_methodology_plot_input.json";compat.write_text(json.dumps(summaries,indent=2),encoding="utf-8")
    plot_methodology_comparison(compat,results/"methodology_comparison.png");compat.unlink(missing_ok=True)
    print(json.dumps(payload,indent=2))
if __name__=="__main__":main()

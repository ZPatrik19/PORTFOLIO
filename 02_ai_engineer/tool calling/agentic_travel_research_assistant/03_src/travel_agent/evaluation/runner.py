from __future__ import annotations
import argparse,csv,json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from travel_agent.agent import build_agent
from travel_agent.config import get_settings
from .metrics import aggregate,score_case

def evaluate_agent(agent:Any,dataset_path:Path,output_dir:Path,limit:int|None=None)->tuple[list[dict],dict]:
    cases=json.loads(dataset_path.read_text(encoding="utf-8"));cases=cases[:limit] if limit else cases
    rows=[];detailed=[]
    for case in cases:
        run=agent.run(case["query"]);calls=[asdict(x) for x in run.trace];scores=score_case(case["expected_calls"],calls,run.answer)
        row={"id":case["id"],**scores,"latency_ms":run.total_latency_ms};rows.append(row)
        detailed.append({"id":case["id"],"query":case["query"],"expected_calls":case["expected_calls"],"actual_calls":calls,"answer":run.answer,"scores":scores,"latency_ms":run.total_latency_ms,"agent_metadata":run.metadata})
    summary=aggregate(rows);output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/"evaluation_details.json").write_text(json.dumps(detailed,ensure_ascii=False,indent=2),encoding="utf-8");(output_dir/"metrics.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    if rows:
        with (output_dir/"case_metrics.csv").open("w",encoding="utf-8",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    return rows,summary

def evaluate(dataset_path:Path,output_dir:Path):return evaluate_agent(build_agent(get_settings()),dataset_path,output_dir)

def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset",default="01_data/benchmark/agent_tasks.json");p.add_argument("--output",default="06_results");p.add_argument("--limit",type=int);a=p.parse_args();_,summary=evaluate_agent(build_agent(get_settings()),Path(a.dataset),Path(a.output),a.limit);print(json.dumps(summary,indent=2))
if __name__=="__main__":main()

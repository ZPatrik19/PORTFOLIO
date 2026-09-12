from __future__ import annotations
import argparse
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for p in [ROOT, SRC]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

STEPS = {
    1: "workflow.step01_problem_and_data",
    2: "workflow.step02_eda",
    3: "workflow.step03_environment_design",
    4: "workflow.step04_baseline_policies",
    5: "workflow.step05_q_learning",
    6: "workflow.step06_dqn",
    7: "workflow.step07_ppo",
    8: "workflow.step08_evaluation_stress_ablation",
    9: "workflow.step09_policy_analysis",
}


def parse_steps(text: str) -> list[int]:
    text = text.strip().lower()
    if text in {"all", "1-9"}:
        return list(STEPS)
    out=[]
    for part in text.split(","):
        part=part.strip()
        if "-" in part:
            a,b=map(int,part.split("-",1)); out.extend(range(a,b+1))
        else:
            out.append(int(part))
    invalid=[x for x in out if x not in STEPS]
    if invalid: raise ValueError(f"Unknown workflow step(s): {invalid}")
    return list(dict.fromkeys(out))


def main():
    parser=argparse.ArgumentParser(description="Run the Smart Battery Energy Management RL workflow.")
    parser.add_argument("--steps", default="all", help="Examples: all, 1-5, 1,3,5, 8-9")
    parser.add_argument("--quick", action="store_true", help="Use short training/evaluation budgets for a fast smoke run.")
    parser.add_argument("--config", default=str(ROOT/"config.yaml"), help="Path to config YAML.")
    args=parser.parse_args()
    steps=parse_steps(args.steps)
    print(f"Running steps: {steps} | quick={args.quick}")
    for n in steps:
        module=importlib.import_module(STEPS[n])
        print(f"\n>>> STEP {n:02d}: {STEPS[n]}")
        module.main(config_path=args.config, quick=args.quick)
    print("\nProject workflow finished. See 04_results/ for generated artifacts.")

if __name__ == "__main__":
    main()

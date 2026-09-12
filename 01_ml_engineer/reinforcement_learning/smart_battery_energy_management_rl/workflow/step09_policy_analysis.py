from __future__ import annotations
import pandas as pd
from workflow.shared.utils import ROOT, ensure_data, load_config, build_env, section
from workflow.shared.visualization import plot_soc_distribution, plot_action_distribution, plot_policy_heatmap
from battery_rl.agents.q_learning import TabularQLearningAgent
from battery_rl.evaluation import evaluate_policy


def main(config_path=None, quick=False):
    section("STEP 09 — Learned policy interpretation")
    cfg=load_config(config_path); _,_,_,test=ensure_data(config_path)
    agent=TabularQLearningAgent.load(ROOT/"04_results/models/q_learning_agent.pkl")
    factory=lambda: build_env(test,cfg,random_start=True)
    metrics,logs=evaluate_policy(factory,agent,seeds=list(cfg["project"]["seeds"]),episodes=(8 if quick else 30))
    logs.to_csv(ROOT/"04_results/predictions/step09_policy_analysis_logs.csv",index=False)
    plot_soc_distribution(logs,ROOT/"04_results/figures/15_soc_distribution.png")
    plot_action_distribution(logs,ROOT/"04_results/figures/16_action_distribution.png")
    plot_policy_heatmap(agent,build_env(test,cfg),ROOT/"04_results/figures/17_learned_policy_heatmap.png")
    # Compact final report table.
    summary=metrics.mean(numeric_only=True).to_frame("mean"); summary["std"]=metrics.std(numeric_only=True); summary.to_csv(ROOT/"04_results/metrics/step09_q_policy_mean_std.csv")
    print(summary.to_string()); return summary

if __name__ == "__main__": main()

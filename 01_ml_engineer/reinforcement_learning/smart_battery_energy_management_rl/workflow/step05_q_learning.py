from __future__ import annotations
import pandas as pd
from workflow.shared.utils import ROOT, ensure_data, load_config, build_env, section
from workflow.shared.visualization import plot_training_curve, plot_multi_seed_curves
from battery_rl.agents.q_learning import StateDiscretizer, TabularQLearningAgent, train_q_learning
from battery_rl.evaluation import evaluate_policy


def make_agent(cfg, seed):
    q = cfg["q_learning"]
    disc = StateDiscretizer(q["soc_bins"], q["price_bins"], q["demand_bins"], q["renewable_bins"], q["hour_bins"], q.get("forecast_bins",5))
    return TabularQLearningAgent(disc, alpha=q["alpha"], gamma=q["gamma"], epsilon=q["epsilon_start"], epsilon_end=q["epsilon_end"], epsilon_decay=q["epsilon_decay"], seed=seed)


def main(config_path=None, quick=False):
    section("STEP 05 — Tabular Q-Learning")
    cfg = load_config(config_path); _, train, val, test = ensure_data(config_path)
    episodes = int(cfg["analysis"]["quick_q_episodes"] if quick else cfg["q_learning"]["episodes"])
    curves={}; agents={}
    for seed in cfg["project"]["seeds"]:
        env=build_env(train,cfg,random_start=True); agent=make_agent(cfg,int(seed))
        returns,_=train_q_learning(env,agent,episodes,seed=int(seed)); curves[int(seed)]=returns; agents[int(seed)]=agent
    primary=agents[int(cfg["project"]["seed"])]
    primary.save(ROOT/"04_results/models/q_learning_agent.pkl")
    pd.DataFrame({f"seed_{s}":v for s,v in curves.items()}).to_csv(ROOT/"04_results/metrics/step05_q_training_curves.csv", index=False)
    plot_training_curve(curves[int(cfg["project"]["seed"])], ROOT/"04_results/figures/10_q_learning_training_curve.png", rolling_window=int(cfg["analysis"]["rolling_window"]))
    plot_multi_seed_curves(curves, ROOT/"04_results/figures/11_q_learning_multi_seed.png")
    val_factory=lambda: build_env(val,cfg,random_start=True)
    val_metrics,_=evaluate_policy(val_factory,primary,seeds=list(cfg["project"]["seeds"]),episodes=(6 if quick else 20))
    val_metrics.to_csv(ROOT/"04_results/metrics/step05_q_learning_validation_metrics.csv",index=False)
    factory=lambda: build_env(test,cfg,random_start=True)
    metrics,logs=evaluate_policy(factory,primary,seeds=list(cfg["project"]["seeds"]),episodes=(8 if quick else int(cfg["analysis"]["evaluation_episodes"])))
    metrics.to_csv(ROOT/"04_results/metrics/step05_q_learning_test_metrics.csv",index=False); logs.to_csv(ROOT/"04_results/predictions/step05_q_learning_logs.csv",index=False)
    summary=metrics.mean(numeric_only=True); print(summary.to_string()); return summary

if __name__ == "__main__": main()

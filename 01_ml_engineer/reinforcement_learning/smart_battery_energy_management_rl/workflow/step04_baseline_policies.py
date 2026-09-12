from __future__ import annotations
import pandas as pd
from workflow.shared.utils import ROOT, ensure_data, load_config, build_env, section
from workflow.shared.visualization import plot_policy_comparison
from battery_rl.agents.baselines import NoBatteryAgent, RandomValidAgent, RuleBasedAgent, PerfectInformationHeuristicAgent
from battery_rl.evaluation import evaluate_policy


def _evaluate(agent, test, cfg, episodes):
    factory = lambda: build_env(test, cfg, random_start=True)
    m, logs = evaluate_policy(factory, agent, seeds=list(cfg["project"]["seeds"]), episodes=episodes)
    row = m.mean(numeric_only=True).to_dict()
    std = m.std(numeric_only=True).to_dict()
    row.update({f"{k}_std": v for k, v in std.items()})
    row["policy"] = agent.name
    logs["policy"] = agent.name
    return row, logs


def main(config_path=None, quick=False):
    section("STEP 04 — Baseline controllers")
    cfg = load_config(config_path); _, train, _, test = ensure_data(config_path)
    episodes = 8 if quick else int(cfg["analysis"]["evaluation_episodes"])
    low = float(train.electricity_price_eur_per_kwh.quantile(0.30)); high = float(train.electricity_price_eur_per_kwh.quantile(0.75))
    agents = [NoBatteryAgent(), RandomValidAgent(seed=42), RuleBasedAgent(low, high), PerfectInformationHeuristicAgent()]
    rows=[]; logs=[]
    for a in agents:
        r,l = _evaluate(a, test, cfg, episodes); rows.append(r); logs.append(l)
    df = pd.DataFrame(rows)
    nb_cost = float(df.loc[df.policy=="No Battery","total_electricity_cost"].iloc[0]); nb_peak=float(df.loc[df.policy=="No Battery","peak_grid_demand_kw"].iloc[0])
    df["cost_saving_pct"] = 100*(nb_cost-df.total_electricity_cost)/nb_cost
    df["peak_reduction_pct"] = 100*(nb_peak-df.peak_grid_demand_kw)/nb_peak
    df.to_csv(ROOT/"04_results/metrics/step04_baseline_comparison.csv", index=False)
    pd.concat(logs, ignore_index=True).to_csv(ROOT/"04_results/predictions/step04_baseline_logs.csv", index=False)
    plot_policy_comparison(df, "total_electricity_cost", ROOT/"04_results/figures/08_baseline_cost_comparison.png", "Baseline electricity cost", "EUR / episode")
    plot_policy_comparison(df, "peak_grid_demand_kw", ROOT/"04_results/figures/09_baseline_peak_comparison.png", "Baseline peak demand", "kW")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__": main()

from __future__ import annotations
import pandas as pd
from workflow.shared.utils import ROOT, ensure_data, load_config, build_env, section
from workflow.shared.visualization import plot_policy_comparison
from battery_rl.data import scenario_transform
from battery_rl.agents.baselines import NoBatteryAgent, RuleBasedAgent
from battery_rl.agents.q_learning import TabularQLearningAgent, train_q_learning
from workflow.step05_q_learning import make_agent
from battery_rl.evaluation import evaluate_policy


def eval_agent(agent, df, cfg, episodes=10, *, include_forecast=False):
    factory=lambda: build_env(df,cfg,random_start=True,include_forecast=include_forecast)
    m,_=evaluate_policy(factory,agent,seeds=list(cfg["project"]["seeds"]),episodes=episodes)
    row = m.mean(numeric_only=True).to_dict()
    row.update({f"{k}_std": v for k, v in m.std(numeric_only=True).to_dict().items()})
    return row


def main(config_path=None, quick=False):
    section("STEP 08 — Evaluation, stress tests and ablations")
    cfg=load_config(config_path); _,train,_,test=ensure_data(config_path); episodes=5 if quick else 15
    low=float(train.electricity_price_eur_per_kwh.quantile(.30)); high=float(train.electricity_price_eur_per_kwh.quantile(.75))
    q_path=ROOT/"04_results/models/q_learning_agent.pkl"
    q_agent=TabularQLearningAgent.load(q_path) if q_path.exists() else make_agent(cfg,42)
    if not q_path.exists(): train_q_learning(build_env(train,cfg),q_agent,80 if quick else 250,seed=42)

    # Consolidated comparison from available saved metrics.
    rows=[]
    policies=[("No Battery",NoBatteryAgent()),("Rule-based",RuleBasedAgent(low,high)),("Q-Learning",q_agent)]
    for name,agent in policies:
        r=eval_agent(agent,test,cfg,episodes); r["policy"]=name; rows.append(r)
    for name,file in [("DQN","step06_dqn_test_metrics.csv"),("PPO","step07_ppo_test_metrics.csv")]:
        p=ROOT/"04_results/metrics"/file
        if p.exists():
            raw=pd.read_csv(p)
            d=raw.mean(numeric_only=True).to_dict()
            d.update({f"{k}_std": v for k, v in raw.std(numeric_only=True).to_dict().items() if k != "training_seed"})
            d.pop("training_seed", None)
            d["policy"]=name; rows.append(d)
    comp=pd.DataFrame(rows)
    nb=comp.loc[comp.policy=="No Battery"].iloc[0]
    comp["cost_saving_pct"]=100*(nb.total_electricity_cost-comp.total_electricity_cost)/nb.total_electricity_cost
    comp["peak_reduction_pct"]=100*(nb.peak_grid_demand_kw-comp.peak_grid_demand_kw)/nb.peak_grid_demand_kw
    comp.to_csv(ROOT/"04_results/metrics/step08_model_comparison.csv",index=False)
    plot_policy_comparison(comp,"total_electricity_cost",ROOT/"04_results/figures/12_policy_cost_comparison.png","Policy electricity cost comparison","EUR / episode")
    plot_policy_comparison(comp,"cost_saving_pct",ROOT/"04_results/figures/13_cost_saving_comparison.png","Cost saving vs no battery","%")
    plot_policy_comparison(comp,"peak_grid_demand_kw",ROOT/"04_results/figures/14_peak_demand_comparison.png","Peak grid demand comparison","kW")

    # Stress tests with fixed learned Q policy.
    stress=[]
    for scenario in ["normal","high_price","volatile_price","high_renewable","low_renewable"]:
        sdf=scenario_transform(test,scenario,seed=2026); r=eval_agent(q_agent,sdf,cfg,episodes); r["scenario"]=scenario; stress.append(r)
    pd.DataFrame(stress).to_csv(ROOT/"04_results/metrics/step08_stress_tests.csv",index=False)

    # Reward ablation: degradation penalty off vs on.
    ablation=[]
    qcfg_episodes=70 if quick else 220
    for label,degr in [("without_degradation_penalty",0.0),("with_degradation_penalty",float(cfg["reward"]["degradation_eur_per_kwh_throughput"]))]:
        agent=make_agent(cfg,123); env=build_env(train,cfg,reward_override={"degradation_eur_per_kwh_throughput":degr})
        train_q_learning(env,agent,qcfg_episodes,seed=123)
        factory=lambda degr=degr: build_env(test,cfg,reward_override={"degradation_eur_per_kwh_throughput":degr})
        m,_=evaluate_policy(factory,agent,seeds=[42,123,2026],episodes=episodes); r=m.mean(numeric_only=True).to_dict(); r["ablation"]=label; ablation.append(r)

    # State ablation: forecast absent/present. Forecast is a simulated decision-time forecast, not future actual leakage.
    for label,forecast in [("state_without_forecast",False),("state_with_forecast",True)]:
        agent=make_agent(cfg,2026); env=build_env(train,cfg,include_forecast=forecast)
        train_q_learning(env,agent,qcfg_episodes,seed=2026)
        factory=lambda forecast=forecast: build_env(test,cfg,include_forecast=forecast)
        m,_=evaluate_policy(factory,agent,seeds=[42,123,2026],episodes=episodes); r=m.mean(numeric_only=True).to_dict(); r["ablation"]=label; ablation.append(r)
    pd.DataFrame(ablation).to_csv(ROOT/"04_results/metrics/step08_ablation_study.csv",index=False)
    print(comp.to_string(index=False)); return comp

if __name__ == "__main__": main()

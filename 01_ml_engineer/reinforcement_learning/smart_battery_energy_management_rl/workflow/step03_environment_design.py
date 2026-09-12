from __future__ import annotations
from workflow.shared.utils import ROOT, ensure_data, load_config, build_env, section, save_json
from workflow.shared.visualization import plot_episode, plot_reward_decomposition, plot_single_episode_series, plot_cumulative_reward


def main(config_path=None, quick=False):
    section("STEP 03 — Gymnasium-compatible BatteryEnvironment")
    cfg = load_config(config_path)
    _, train, _, _ = ensure_data(config_path)
    env = build_env(train, cfg, random_start=False)
    obs, info = env.reset(seed=42, options={"start_index": 0})
    # Educational sequence deliberately requests impossible charging near max SOC.
    for t in range(env.episode_hours):
        price = float(env._row()["electricity_price_eur_per_kwh"])
        action = 0 if price < train.electricity_price_eur_per_kwh.quantile(0.30) else (2 if price > train.electricity_price_eur_per_kwh.quantile(0.75) else 1)
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc: break
    log = env.episode_dataframe()
    log.to_csv(ROOT/"04_results/predictions/step03_environment_episode.csv", index=False)
    plot_episode(log, ROOT/"04_results/figures/06_environment_episode.png")
    plot_reward_decomposition(log, ROOT/"04_results/figures/07_reward_decomposition.png")
    plot_single_episode_series(log, "soc", ROOT/"04_results/figures/06a_battery_soc.png", "Battery state of charge", "SOC")
    plot_single_episode_series(log, "executed_battery_power_kw", ROOT/"04_results/figures/06b_charge_discharge_action.png", "Executed charge/discharge action", "Battery power [kW]")
    plot_single_episode_series(log, "grid_import_kw", ROOT/"04_results/figures/06c_grid_demand.png", "Grid import", "Grid import [kW]")
    plot_cumulative_reward(log, ROOT/"04_results/figures/07a_cumulative_reward.png")
    summary = {
        "initial_soc": cfg["battery"]["initial_soc"],
        "final_soc": float(log.soc.iloc[-1]),
        "requested_executed_mismatch_events": int(log.constraint_violation.sum()),
        "episode_return": float(log.reward.sum()),
        "energy_balance_definition": "grid = demand - renewable + charge - discharge",
    }
    save_json(summary, "04_results/metrics/step03_environment_summary.json")
    print(summary)
    return summary

if __name__ == "__main__": main()

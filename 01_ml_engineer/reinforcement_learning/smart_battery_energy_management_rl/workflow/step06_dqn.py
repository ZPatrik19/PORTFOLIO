from __future__ import annotations

import pandas as pd

from workflow.shared.utils import ROOT, ensure_data, load_config, build_env, section
from battery_rl.agents.dqn_torch import TorchDQNAgent, train_dqn
from battery_rl.evaluation import evaluate_policy


def main(config_path=None, quick=False):
    section("STEP 06 — Deep Q-Network (educational PyTorch implementation)")
    cfg = load_config(config_path)
    _, train, val, test = ensure_data(config_path)
    dcfg = cfg.get("dqn", {})
    total_steps = int(dcfg.get("quick_timesteps", 4000) if quick else dcfg.get("timesteps", 30000))
    model_seeds = [int(cfg["project"]["seed"])] if quick else [int(s) for s in cfg["project"]["seeds"]]
    val_rows, test_rows = [], []

    for seed in model_seeds:
        env = build_env(train, cfg, random_start=True)
        agent = TorchDQNAgent(
            observation_dim=len(env.obs_columns),
            learning_rate=float(dcfg.get("learning_rate", 3e-4)),
            gamma=float(dcfg.get("gamma", 0.97)),
            buffer_size=int(dcfg.get("buffer_size", 50000)),
            batch_size=int(dcfg.get("batch_size", 64)),
            target_update_interval=int(dcfg.get("target_update_interval", 250)),
            epsilon_start=float(dcfg.get("epsilon_start", 1.0)),
            epsilon_end=float(dcfg.get("epsilon_end", 0.05)),
            epsilon_decay_steps=int(dcfg.get("epsilon_decay_steps", max(1000, total_steps // 2))),
            seed=seed,
        )
        history = train_dqn(
            env,
            agent,
            total_steps=total_steps,
            learning_starts=int(dcfg.get("learning_starts", min(500, max(64, total_steps // 10)))),
            train_frequency=int(dcfg.get("train_frequency", 1)),
            seed=seed,
        )
        agent.save(ROOT / f"04_results/models/dqn_torch_seed_{seed}.pt")
        hist = pd.DataFrame({"episode_return": history.episode_returns, "episode_cost": history.episode_costs})
        hist.to_csv(ROOT / f"04_results/metrics/step06_dqn_training_seed_{seed}.csv", index=False)
        pd.DataFrame({"loss": history.losses, "mean_q": history.q_values}).to_csv(
            ROOT / f"04_results/metrics/step06_dqn_updates_seed_{seed}.csv", index=False
        )

        n_eval = 5 if quick else 15
        vm, _ = evaluate_policy(lambda: build_env(val, cfg, random_start=True), agent, seeds=[42, 123, 2026], episodes=n_eval)
        tm, logs = evaluate_policy(lambda: build_env(test, cfg, random_start=True), agent, seeds=[42, 123, 2026], episodes=n_eval)
        v = vm.mean(numeric_only=True).to_dict(); v["training_seed"] = seed; val_rows.append(v)
        t = tm.mean(numeric_only=True).to_dict(); t["training_seed"] = seed; test_rows.append(t)
        logs.to_csv(ROOT / f"04_results/predictions/step06_dqn_seed_{seed}_logs.csv", index=False)

    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)
    val_df.to_csv(ROOT / "04_results/metrics/step06_dqn_validation_metrics.csv", index=False)
    test_df.to_csv(ROOT / "04_results/metrics/step06_dqn_test_metrics.csv", index=False)
    summary = pd.DataFrame({"mean": test_df.mean(numeric_only=True), "std": test_df.std(numeric_only=True)}).drop(index="training_seed", errors="ignore")
    summary.to_csv(ROOT / "04_results/metrics/step06_dqn_mean_std.csv")
    print(summary.to_string())
    return test_df


if __name__ == "__main__":
    main()

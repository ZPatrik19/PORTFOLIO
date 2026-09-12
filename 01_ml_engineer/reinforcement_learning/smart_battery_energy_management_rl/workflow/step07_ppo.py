from __future__ import annotations

import pandas as pd

from workflow.shared.utils import ROOT, ensure_data, load_config, build_env, section
from battery_rl.agents.ppo_torch import TorchPPOAgent, train_ppo
from battery_rl.evaluation import evaluate_policy


def main(config_path=None, quick=False):
    section("STEP 07 — PPO with continuous battery control (educational PyTorch implementation)")
    cfg = load_config(config_path)
    _, train, val, test = ensure_data(config_path)
    pcfg = cfg.get("ppo", {})
    total_steps = int(pcfg.get("quick_timesteps", 4000) if quick else pcfg.get("timesteps", 30000))
    model_seeds = [int(cfg["project"]["seed"])] if quick else [int(s) for s in cfg["project"]["seeds"]]
    val_rows, test_rows = [], []

    for seed in model_seeds:
        env = build_env(train, cfg, continuous=True, random_start=True)
        agent = TorchPPOAgent(
            observation_dim=len(env.obs_columns),
            learning_rate=float(pcfg.get("learning_rate", 3e-4)),
            gamma=float(pcfg.get("gamma", 0.97)),
            gae_lambda=float(pcfg.get("gae_lambda", 0.95)),
            clip_coef=float(pcfg.get("clip_coef", 0.2)),
            value_coef=float(pcfg.get("value_coef", 0.5)),
            entropy_coef=float(pcfg.get("entropy_coef", 0.01)),
            update_epochs=int(pcfg.get("update_epochs", 6)),
            minibatch_size=int(pcfg.get("minibatch_size", 64)),
            seed=seed,
        )
        history = train_ppo(
            env,
            agent,
            total_timesteps=total_steps,
            rollout_steps=int(pcfg.get("rollout_steps", 512)),
            seed=seed,
        )
        agent.save(ROOT / f"04_results/models/ppo_torch_seed_{seed}.pt")
        pd.DataFrame({"episode_return": history.episode_returns}).to_csv(
            ROOT / f"04_results/metrics/step07_ppo_training_seed_{seed}.csv", index=False
        )
        pd.DataFrame({
            "policy_loss": history.policy_losses,
            "value_loss": history.value_losses,
            "entropy": history.entropies,
            "approx_kl": history.approx_kls,
        }).to_csv(ROOT / f"04_results/metrics/step07_ppo_updates_seed_{seed}.csv", index=False)

        n_eval = 5 if quick else 15
        vm, _ = evaluate_policy(lambda: build_env(val, cfg, continuous=True, random_start=True), agent, seeds=[42, 123, 2026], episodes=n_eval)
        tm, logs = evaluate_policy(lambda: build_env(test, cfg, continuous=True, random_start=True), agent, seeds=[42, 123, 2026], episodes=n_eval)
        v = vm.mean(numeric_only=True).to_dict(); v["training_seed"] = seed; val_rows.append(v)
        t = tm.mean(numeric_only=True).to_dict(); t["training_seed"] = seed; test_rows.append(t)
        logs.to_csv(ROOT / f"04_results/predictions/step07_ppo_seed_{seed}_logs.csv", index=False)

    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)
    val_df.to_csv(ROOT / "04_results/metrics/step07_ppo_validation_metrics.csv", index=False)
    test_df.to_csv(ROOT / "04_results/metrics/step07_ppo_test_metrics.csv", index=False)
    summary = pd.DataFrame({"mean": test_df.mean(numeric_only=True), "std": test_df.std(numeric_only=True)}).drop(index="training_seed", errors="ignore")
    summary.to_csv(ROOT / "04_results/metrics/step07_ppo_mean_std.csv")
    print(summary.to_string())
    return test_df


if __name__ == "__main__":
    main()

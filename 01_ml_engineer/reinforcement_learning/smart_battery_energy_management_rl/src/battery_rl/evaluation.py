from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any
import numpy as np
import pandas as pd


@dataclass
class EpisodeMetrics:
    total_electricity_cost: float
    peak_grid_demand_kw: float
    renewable_self_consumption_pct: float
    battery_throughput_kwh: float
    battery_cycles: float
    constraint_violations: int
    average_soc: float
    cumulative_reward: float
    episode_return: float


def metrics_from_log(log: pd.DataFrame, capacity_kwh: float) -> EpisodeMetrics:
    renewable_total = float(log["renewable_generation_kwh"].sum())
    renewable_used = float(log["renewable_used_locally_kwh"].sum())
    throughput = float(log["throughput_kwh"].sum())
    ret = float(log["reward"].sum())
    return EpisodeMetrics(
        total_electricity_cost=float(log["electricity_cost"].sum()),
        peak_grid_demand_kw=float(log["grid_import_kw"].max()),
        renewable_self_consumption_pct=100.0 * renewable_used / max(renewable_total, 1e-9),
        battery_throughput_kwh=throughput,
        battery_cycles=throughput / max(2.0 * capacity_kwh, 1e-9),
        constraint_violations=int(log["constraint_violation"].sum()),
        average_soc=float(log["soc"].mean()),
        cumulative_reward=ret,
        episode_return=ret,
    )


def run_episode(env, policy: Any, *, seed: int = 42, start_index: int | None = None):
    options = {} if start_index is None else {"start_index": start_index}
    obs, _ = env.reset(seed=seed, options=options)
    while True:
        if hasattr(policy, "act"):
            action = policy.act(obs, env)
        else:
            action = policy(obs, env)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    log = env.episode_dataframe()
    return metrics_from_log(log, env.spec.capacity_kwh), log


def evaluate_policy(env_factory: Callable[[], Any], policy: Any, *, seeds: list[int], episodes: int = 20):
    rows = []
    logs = []
    for i in range(episodes):
        seed = seeds[i % len(seeds)] + i * 1009
        env = env_factory()
        m, log = run_episode(env, policy, seed=seed)
        rows.append(m.__dict__)
        log = log.copy()
        log["episode"] = i
        logs.append(log)
    metrics = pd.DataFrame(rows)
    return metrics, pd.concat(logs, ignore_index=True)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"mean": metrics.mean(numeric_only=True), "std": metrics.std(numeric_only=True)}).reset_index(names="metric")


def add_savings_columns(summary: pd.DataFrame, no_battery_cost: float, no_battery_peak: float) -> pd.DataFrame:
    out = summary.copy()
    out["cost_saving_pct"] = 100.0 * (no_battery_cost - out["total_electricity_cost"]) / max(no_battery_cost, 1e-9)
    out["peak_reduction_pct"] = 100.0 * (no_battery_peak - out["peak_grid_demand_kw"]) / max(no_battery_peak, 1e-9)
    return out

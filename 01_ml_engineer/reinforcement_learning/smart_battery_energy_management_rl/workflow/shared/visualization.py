from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path: str | Path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_timeseries(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, path):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df[x], df[y], linewidth=1.2)
    ax.set_title(title); ax.set_xlabel("Time"); ax.set_ylabel(ylabel); ax.grid(alpha=0.25)
    return _save(fig, path)


def plot_episode(log: pd.DataFrame, path):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    x = np.arange(len(log))
    axes[0].plot(x, log["price"]); axes[0].set_ylabel("EUR/kWh"); axes[0].set_title("Episode: price, SOC, action and grid import")
    axes[1].plot(x, log["soc"]); axes[1].set_ylabel("SOC")
    axes[2].step(x, log["executed_battery_power_kw"], where="mid"); axes[2].axhline(0, linewidth=0.8); axes[2].set_ylabel("Battery kW")
    axes[3].plot(x, log["grid_import_kw"], label="Grid import"); axes[3].plot(x, log["demand_kw"], label="Demand", alpha=0.8); axes[3].set_ylabel("kW"); axes[3].set_xlabel("Hour"); axes[3].legend()
    for ax in axes: ax.grid(alpha=0.2)
    return _save(fig, path)


def plot_reward_decomposition(log: pd.DataFrame, path):
    comp = pd.DataFrame({
        "electricity_cost": -log["electricity_cost"],
        "peak_penalty": -log["peak_penalty"],
        "degradation_penalty": -log["degradation_penalty"],
        "constraint_penalty": -log["constraint_penalty"],
    })
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.stackplot(np.arange(len(comp)), *[comp[c].to_numpy() for c in comp.columns], labels=comp.columns, alpha=0.8)
    ax.set_title("Reward decomposition by timestep"); ax.set_xlabel("Hour"); ax.set_ylabel("Reward contribution"); ax.legend(loc="best"); ax.grid(alpha=0.2)
    return _save(fig, path)


def plot_training_curve(returns, path, title="Episode return training curve", rolling_window=30):
    s = pd.Series(np.asarray(returns, dtype=float))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(s.index, s, alpha=0.35, label="Episode return")
    ax.plot(s.index, s.rolling(rolling_window, min_periods=1).mean(), linewidth=2, label=f"Rolling mean ({rolling_window})")
    ax.set_title(title); ax.set_xlabel("Episode"); ax.set_ylabel("Return"); ax.legend(); ax.grid(alpha=0.25)
    return _save(fig, path)


def plot_multi_seed_curves(curves: dict[int, np.ndarray], path, rolling_window=20):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for seed, values in curves.items():
        s = pd.Series(values)
        ax.plot(s.rolling(rolling_window, min_periods=1).mean(), label=f"seed={seed}")
    ax.set_title("Multiple-seed Q-learning training curves"); ax.set_xlabel("Episode"); ax.set_ylabel("Rolling episode return"); ax.legend(); ax.grid(alpha=0.25)
    return _save(fig, path)


def plot_policy_comparison(df: pd.DataFrame, metric: str, path, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    ax.bar(x, df[metric].to_numpy())
    ax.set_xticks(x, df["policy"], rotation=20, ha="right")
    ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.25)
    return _save(fig, path)


def plot_soc_distribution(logs: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(logs["soc"], bins=25, alpha=0.85)
    ax.set_title("SOC distribution"); ax.set_xlabel("State of charge"); ax.set_ylabel("Frequency"); ax.grid(axis="y", alpha=0.25)
    return _save(fig, path)


def plot_action_distribution(logs: pd.DataFrame, path):
    vals = logs["executed_battery_power_kw"]
    labels = ["discharge", "idle", "charge"]
    counts = [(vals < -1e-6).sum(), (vals.abs() <= 1e-6).sum(), (vals > 1e-6).sum()]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, counts)
    ax.set_title("Action distribution (executed actions)"); ax.set_ylabel("Count"); ax.grid(axis="y", alpha=0.25)
    return _save(fig, path)



def plot_single_episode_series(log: pd.DataFrame, column: str, path, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(log))
    if column == "executed_battery_power_kw":
        ax.step(x, log[column], where="mid")
        ax.axhline(0, linewidth=0.8)
    else:
        ax.plot(x, log[column])
    ax.set_title(title); ax.set_xlabel("Hour"); ax.set_ylabel(ylabel); ax.grid(alpha=0.25)
    return _save(fig, path)

def plot_cumulative_reward(log: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    cumulative = log["reward"].cumsum()
    ax.plot(np.arange(len(log)), cumulative)
    ax.set_title("Cumulative reward within episode"); ax.set_xlabel("Hour"); ax.set_ylabel("Cumulative reward"); ax.grid(alpha=0.25)
    return _save(fig, path)

def plot_policy_heatmap(agent, env, path, hour=18, demand_norm=0.55, renewable_norm=0.15):
    socs = np.linspace(0.1, 0.9, 17)
    prices = np.linspace(0.05, 0.30, 26)
    mat = np.zeros((len(socs), len(prices)), dtype=int)
    for i, soc in enumerate(socs):
        for j, price in enumerate(prices):
            raw = np.array([soc, price, demand_norm * env._demand_max, renewable_norm * env._renewable_max, hour], dtype=np.float32)
            scale = np.array([1.0, env._price_max, env._demand_max, env._renewable_max, 23.0], dtype=np.float32)
            obs = raw / scale
            state = agent.discretizer.transform(obs)
            q = agent._values(state)
            valid = [1]
            if soc < env.spec.max_soc - 1e-8:
                valid.append(0)
            if soc > env.spec.min_soc + 1e-8:
                valid.append(2)
            mat[i, j] = int(max(valid, key=lambda a: q[a]))
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(mat, aspect="auto", origin="lower", extent=[prices.min(), prices.max(), socs.min(), socs.max()], vmin=0, vmax=2)
    ax.set_title("Learned policy heatmap: electricity price × battery SOC"); ax.set_xlabel("Electricity price [EUR/kWh]"); ax.set_ylabel("SOC")
    cbar = fig.colorbar(im, ax=ax, ticks=[0,1,2]); cbar.ax.set_yticklabels(["charge", "idle", "discharge"])
    return _save(fig, path)

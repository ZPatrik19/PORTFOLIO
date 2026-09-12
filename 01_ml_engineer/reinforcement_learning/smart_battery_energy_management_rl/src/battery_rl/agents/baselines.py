from __future__ import annotations
import numpy as np


class NoBatteryAgent:
    name = "No Battery"
    def act(self, obs, env) -> int:
        return 1


class RandomValidAgent:
    name = "Random"
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
    def act(self, obs, env) -> int:
        return int(self.rng.choice(env.valid_discrete_actions()))


class RuleBasedAgent:
    name = "Rule-based"
    def __init__(self, low_price: float, high_price: float):
        self.low_price = float(low_price)
        self.high_price = float(high_price)
    def act(self, obs, env) -> int:
        price = float(env._row()["electricity_price_eur_per_kwh"])
        if price <= self.low_price and 0 in env.valid_discrete_actions():
            return 0
        if price >= self.high_price and 2 in env.valid_discrete_actions():
            return 2
        return 1


class PerfectInformationHeuristicAgent:
    """Daily oracle-style heuristic used as an interpretable upper-reference.

    It is *not* a fair online controller because it sees the episode's full price
    profile. It is intentionally labeled as perfect-information.
    """
    name = "Perfect-info heuristic"
    def act(self, obs, env) -> int:
        start = env.start_index
        end = min(start + env.episode_hours, len(env.data))
        prices = env.data.iloc[start:end]["electricity_price_eur_per_kwh"]
        low = float(prices.quantile(0.25))
        high = float(prices.quantile(0.75))
        price = float(env._row()["electricity_price_eur_per_kwh"])
        if price <= low and 0 in env.valid_discrete_actions():
            return 0
        if price >= high and 2 in env.valid_discrete_actions():
            return 2
        return 1

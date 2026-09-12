from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pickle
import numpy as np


@dataclass
class StateDiscretizer:
    soc_bins: int = 8
    price_bins: int = 8
    demand_bins: int = 6
    renewable_bins: int = 6
    hour_bins: int = 6
    forecast_bins: int = 5

    def transform(self, normalized_observation: np.ndarray) -> tuple[int, ...]:
        obs = np.asarray(normalized_observation, dtype=float)
        bins = [self.soc_bins, self.price_bins, self.demand_bins, self.renewable_bins, self.hour_bins]
        if len(obs) > 5:
            bins += [self.forecast_bins] * (len(obs) - 5)
        state = []
        for value, n_bins in zip(obs, bins):
            idx = int(np.floor(np.clip(value, 0.0, 0.999999) * n_bins))
            state.append(min(max(idx, 0), n_bins - 1))
        return tuple(state)


class TabularQLearningAgent:
    def __init__(
        self,
        discretizer: StateDiscretizer,
        *,
        alpha: float = 0.12,
        gamma: float = 0.97,
        epsilon: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.992,
        seed: int = 42,
    ) -> None:
        self.discretizer = discretizer
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.q: dict[tuple[int, ...], np.ndarray] = {}

    def _values(self, state: tuple[int, ...]) -> np.ndarray:
        if state not in self.q:
            self.q[state] = np.zeros(3, dtype=np.float64)
        return self.q[state]

    def choose_action(self, obs: np.ndarray, valid_actions: list[int], explore: bool = True) -> int:
        state = self.discretizer.transform(obs)
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid_actions))
        q = self._values(state)
        valid_q = np.array([q[a] for a in valid_actions])
        best = np.flatnonzero(valid_q == valid_q.max())
        return int(valid_actions[int(self.rng.choice(best))])

    def update(self, obs, action, reward, next_obs, done, next_valid_actions):
        s = self.discretizer.transform(obs)
        ns = self.discretizer.transform(next_obs)
        q_sa = self._values(s)[action]
        next_max = 0.0 if done else max(self._values(ns)[a] for a in next_valid_actions)
        target = reward + self.gamma * next_max
        self._values(s)[action] = q_sa + self.alpha * (target - q_sa)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def act(self, obs, env) -> int:
        return self.choose_action(obs, env.valid_discrete_actions(), explore=False)

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path):
        with Path(path).open("rb") as f:
            return pickle.load(f)


def train_q_learning(env, agent: TabularQLearningAgent, episodes: int, seed: int = 42):
    returns = []
    epsilons = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total = 0.0
        while True:
            action = agent.choose_action(obs, env.valid_discrete_actions(), explore=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_valid = env.valid_discrete_actions() if not done else [1]
            agent.update(obs, action, reward, next_obs, done, next_valid)
            obs = next_obs
            total += reward
            if done:
                break
        returns.append(total)
        epsilons.append(agent.epsilon)
        agent.decay_epsilon()
    return np.asarray(returns, dtype=float), np.asarray(epsilons, dtype=float)

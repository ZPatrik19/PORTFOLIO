from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    """Small MLP used to approximate Q(s, a) for the three discrete actions."""

    def __init__(self, observation_dim: int, action_dim: int = 3, hidden_sizes=(128, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = observation_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DQNTrainHistory:
    episode_returns: list[float]
    episode_costs: list[float]
    losses: list[float]
    epsilons: list[float]
    q_values: list[float]


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000, seed: int = 42) -> None:
        self.buffer = deque(maxlen=int(capacity))
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, obs, action, reward, next_obs, done) -> None:
        self.buffer.append((np.asarray(obs, dtype=np.float32), int(action), float(reward), np.asarray(next_obs, dtype=np.float32), float(done)))

    def sample(self, batch_size: int, device: torch.device):
        batch = self.rng.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.tensor(np.stack(obs), dtype=torch.float32, device=device),
            torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(np.stack(next_obs), dtype=torch.float32, device=device),
            torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1),
        )


class TorchDQNAgent:
    """Educational DQN implementation used by the notebook and as an offline fallback.

    The project still keeps Stable-Baselines3 as an optional production-style reference,
    but this implementation makes the learning mechanics inspectable in the repository.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int = 3,
        *,
        learning_rate: float = 3e-4,
        gamma: float = 0.97,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        target_update_interval: int = 250,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10_000,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update_interval = int(target_update_interval)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = max(1, int(epsilon_decay_steps))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.online = QNetwork(observation_dim, action_dim).to(self.device)
        self.target = QNetwork(observation_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=float(learning_rate))
        self.replay = ReplayBuffer(buffer_size, seed=seed)
        self.steps = 0

    def epsilon(self) -> float:
        frac = min(1.0, self.steps / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def act(self, obs, env=None, explore: bool = False) -> int:
        valid = list(range(self.action_dim))
        if env is not None and hasattr(env, "valid_discrete_actions"):
            valid = env.valid_discrete_actions()
        if explore and self.rng.random() < self.epsilon():
            return int(self.rng.choice(valid))
        x = torch.tensor(np.asarray(obs, dtype=np.float32), device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.online(x).squeeze(0).cpu().numpy()
        masked = np.full_like(q, -np.inf, dtype=float)
        masked[valid] = q[valid]
        return int(np.argmax(masked))

    def update(self) -> tuple[float, float] | None:
        if len(self.replay) < self.batch_size:
            return None
        obs, actions, rewards, next_obs, dones = self.replay.sample(self.batch_size, self.device)
        q_sa = self.online(obs).gather(1, actions)
        with torch.no_grad():
            next_q = self.target(next_obs).max(dim=1, keepdim=True).values
            target = rewards + self.gamma * (1.0 - dones) * next_q
        loss = nn.functional.smooth_l1_loss(q_sa, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()
        if self.steps % self.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item()), float(q_sa.detach().mean().item())

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "steps": self.steps,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
        }, p)


def train_dqn(
    env,
    agent: TorchDQNAgent,
    *,
    total_steps: int = 10_000,
    learning_starts: int = 500,
    train_frequency: int = 1,
    seed: int = 42,
) -> DQNTrainHistory:
    history = DQNTrainHistory([], [], [], [], [])
    obs, _ = env.reset(seed=seed)
    episode_return = 0.0
    episode_cost = 0.0

    for step in range(int(total_steps)):
        agent.steps = step
        action = agent.act(obs, env, explore=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        agent.replay.add(obs, action, reward, next_obs, done)
        episode_return += float(reward)
        episode_cost += float(info.get("electricity_cost", 0.0))
        obs = next_obs

        if step >= learning_starts and step % train_frequency == 0:
            result = agent.update()
            if result is not None:
                loss, q_mean = result
                history.losses.append(loss)
                history.q_values.append(q_mean)
        history.epsilons.append(agent.epsilon())

        if done:
            history.episode_returns.append(episode_return)
            history.episode_costs.append(episode_cost)
            obs, _ = env.reset(seed=seed + len(history.episode_returns))
            episode_return = 0.0
            episode_cost = 0.0

    return history

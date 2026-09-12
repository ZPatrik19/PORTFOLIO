from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


class ContinuousActorCritic(nn.Module):
    """Actor-critic with a squashed Gaussian policy for actions in [-1, 1]."""

    def __init__(self, observation_dim: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.actor_body = nn.Sequential(
            nn.Linear(observation_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.tensor([-0.5], dtype=torch.float32))
        self.critic = nn.Sequential(
            nn.Linear(observation_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor_mean(self.actor_body(obs))
        std = self.log_std.clamp(-4.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(-0.999999, 0.999999)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        dist = self.distribution(obs)
        if action is None:
            z = dist.rsample()
            action = torch.tanh(z)
        else:
            z = self._atanh(action)
        log_prob = dist.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        entropy = dist.entropy()
        return action, log_prob.sum(-1), entropy.sum(-1), self.value(obs)


@dataclass
class PPOTrainHistory:
    episode_returns: list[float]
    policy_losses: list[float]
    value_losses: list[float]
    entropies: list[float]
    approx_kls: list[float]


class TorchPPOAgent:
    def __init__(
        self,
        observation_dim: int,
        *,
        learning_rate: float = 3e-4,
        gamma: float = 0.97,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        update_epochs: int = 6,
        minibatch_size: int = 64,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = ContinuousActorCritic(observation_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(learning_rate), eps=1e-5)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_coef = float(clip_coef)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.update_epochs = int(update_epochs)
        self.minibatch_size = int(minibatch_size)

    def act(self, obs, env=None, deterministic: bool = True):
        x = torch.tensor(np.asarray(obs, dtype=np.float32), device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mean = self.model.actor_mean(self.model.actor_body(x))
                action = torch.tanh(mean)
            else:
                action, _, _, _ = self.model.action_and_value(x)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict()}, p)


def train_ppo(
    env,
    agent: TorchPPOAgent,
    *,
    total_timesteps: int = 10_000,
    rollout_steps: int = 512,
    seed: int = 42,
) -> PPOTrainHistory:
    history = PPOTrainHistory([], [], [], [], [])
    device = agent.device
    obs_np, _ = env.reset(seed=seed)
    episode_return = 0.0
    global_step = 0

    while global_step < total_timesteps:
        n = min(rollout_steps, total_timesteps - global_step)
        obs_buf = torch.zeros((n, len(obs_np)), dtype=torch.float32, device=device)
        actions_buf = torch.zeros((n, 1), dtype=torch.float32, device=device)
        logp_buf = torch.zeros(n, dtype=torch.float32, device=device)
        rewards_buf = torch.zeros(n, dtype=torch.float32, device=device)
        dones_buf = torch.zeros(n, dtype=torch.float32, device=device)
        values_buf = torch.zeros(n, dtype=torch.float32, device=device)

        for t in range(n):
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
            obs_buf[t] = obs_t
            with torch.no_grad():
                action, logp, _, value = agent.model.action_and_value(obs_t.unsqueeze(0))
            action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = bool(terminated or truncated)
            actions_buf[t] = action.squeeze(0)
            logp_buf[t] = logp.item()
            rewards_buf[t] = float(reward)
            dones_buf[t] = float(done)
            values_buf[t] = value.item()
            episode_return += float(reward)
            global_step += 1
            obs_np = next_obs
            if done:
                history.episode_returns.append(episode_return)
                episode_return = 0.0
                obs_np, _ = env.reset(seed=seed + len(history.episode_returns))

        with torch.no_grad():
            next_value = agent.model.value(torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)).item()

        advantages = torch.zeros(n, dtype=torch.float32, device=device)
        lastgaelam = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_non_terminal = 1.0 - dones_buf[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones_buf[t]
                next_values = values_buf[t + 1]
            delta = rewards_buf[t] + agent.gamma * next_values * next_non_terminal - values_buf[t]
            lastgaelam = delta + agent.gamma * agent.gae_lambda * next_non_terminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values_buf
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        inds = np.arange(n)
        for _ in range(agent.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, n, agent.minibatch_size):
                mb = torch.tensor(inds[start:start + agent.minibatch_size], dtype=torch.long, device=device)
                _, new_logp, entropy, new_value = agent.model.action_and_value(obs_buf[mb], actions_buf[mb])
                log_ratio = new_logp - logp_buf[mb]
                ratio = log_ratio.exp()
                pg_loss1 = -advantages[mb] * ratio
                pg_loss2 = -advantages[mb] * torch.clamp(ratio, 1 - agent.clip_coef, 1 + agent.clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                value_loss = 0.5 * (new_value - returns[mb]).pow(2).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss + agent.value_coef * value_loss - agent.entropy_coef * entropy_loss
                agent.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.model.parameters(), 0.5)
                agent.optimizer.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                history.policy_losses.append(float(policy_loss.item()))
                history.value_losses.append(float(value_loss.item()))
                history.entropies.append(float(entropy_loss.item()))
                history.approx_kls.append(float(approx_kl))

    return history

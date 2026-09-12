import numpy as np
import torch

from battery_rl.agents.dqn_torch import QNetwork, ReplayBuffer, TorchDQNAgent
from battery_rl.agents.ppo_torch import ContinuousActorCritic, TorchPPOAgent


def test_q_network_output_shape():
    net = QNetwork(observation_dim=5, action_dim=3)
    x = torch.zeros((4, 5), dtype=torch.float32)
    y = net(x)
    assert y.shape == (4, 3)


def test_replay_buffer_batch_shapes():
    rb = ReplayBuffer(capacity=20, seed=42)
    for i in range(8):
        obs = np.full(5, i, dtype=np.float32)
        rb.add(obs, i % 3, float(i), obs + 1, False)
    batch = rb.sample(4, torch.device("cpu"))
    obs, actions, rewards, next_obs, dones = batch
    assert obs.shape == (4, 5)
    assert actions.shape == (4, 1)
    assert rewards.shape == (4, 1)
    assert next_obs.shape == (4, 5)
    assert dones.shape == (4, 1)


def test_dqn_respects_valid_action_mask():
    class DummyEnv:
        def valid_discrete_actions(self):
            return [1]
    agent = TorchDQNAgent(observation_dim=5, seed=42)
    action = agent.act(np.zeros(5, dtype=np.float32), DummyEnv(), explore=False)
    assert action == 1


def test_ppo_deterministic_action_is_bounded():
    agent = TorchPPOAgent(observation_dim=5, seed=42)
    action = agent.act(np.zeros(5, dtype=np.float32), deterministic=True)
    assert action.shape == (1,)
    assert -1.0 <= float(action[0]) <= 1.0


def test_actor_critic_shapes():
    model = ContinuousActorCritic(observation_dim=5)
    x = torch.zeros((3, 5), dtype=torch.float32)
    action, logp, entropy, value = model.action_and_value(x)
    assert action.shape == (3, 1)
    assert logp.shape == (3,)
    assert entropy.shape == (3,)
    assert value.shape == (3,)

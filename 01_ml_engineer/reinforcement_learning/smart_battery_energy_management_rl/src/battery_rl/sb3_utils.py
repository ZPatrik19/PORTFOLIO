from __future__ import annotations
from pathlib import Path
import numpy as np


def import_sb3():
    try:
        from stable_baselines3 import DQN, PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.callbacks import BaseCallback
        return DQN, PPO, Monitor, BaseCallback
    except ImportError as exc:
        raise ImportError(
            "Stable-Baselines3 is not installed. Run `pip install -r requirements.txt` "
            "or `pip install -e .[rl,dev]`."
        ) from exc


def make_predict_policy(model, continuous: bool = False):
    def policy(obs, env):
        action, _ = model.predict(obs, deterministic=True)
        if continuous:
            return np.asarray(action, dtype=np.float32)
        return int(np.asarray(action).item())
    return policy

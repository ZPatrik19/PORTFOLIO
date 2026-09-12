from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # Lightweight fallback lets educational baselines run before pip install.
    class _Env:
        metadata = {}
        np_random: np.random.Generator
        def reset(self, *, seed=None, options=None):
            self.np_random = np.random.default_rng(seed)
    class _Discrete:
        def __init__(self, n): self.n = n
        def sample(self): return int(np.random.randint(self.n))
    class _Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.low = np.array(low, dtype=dtype) if shape is None else np.full(shape, low, dtype=dtype)
            self.high = np.array(high, dtype=dtype) if shape is None else np.full(shape, high, dtype=dtype)
            self.shape = self.low.shape
            self.dtype = dtype
        def sample(self): return np.random.uniform(self.low, self.high).astype(self.dtype)
    class _Spaces:
        Discrete = _Discrete
        Box = _Box
    class _Gym:
        Env = _Env
    gym, spaces = _Gym(), _Spaces()

from battery_rl.rewards import calculate_reward


@dataclass(frozen=True)
class BatterySpec:
    capacity_kwh: float
    initial_soc: float
    min_soc: float
    max_soc: float
    max_charge_kw: float
    max_discharge_kw: float
    charging_efficiency: float
    discharging_efficiency: float
    timestep_hours: float
    feed_in_tariff_eur_per_kwh: float


class BatteryEnvironment(gym.Env):
    """Gymnasium-compatible battery energy management environment.

    Action convention:
      discrete 0 -> charge, 1 -> idle, 2 -> discharge
      continuous +1 -> max charge, 0 -> idle, -1 -> max discharge

    Internally, requested_battery_power_kw > 0 means charging from the AC bus,
    while < 0 means discharging to the AC bus.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        battery: dict[str, float],
        reward_config: dict[str, float],
        *,
        episode_hours: int = 24,
        random_start: bool = True,
        continuous_action: bool = False,
        normalize_observation: bool = True,
        include_forecast: bool = False,
        peak_threshold_kw: float = 70.0,
    ) -> None:
        super().__init__()
        if len(data) < episode_hours:
            raise ValueError("Data must contain at least episode_hours rows.")
        self.data = data.reset_index(drop=True).copy()
        self.spec = BatterySpec(**battery)
        self.reward_config = reward_config.copy()
        self.episode_hours = int(episode_hours)
        self.random_start = bool(random_start)
        self.continuous_action = bool(continuous_action)
        self.normalize_observation = bool(normalize_observation)
        self.include_forecast = bool(include_forecast)
        self.peak_threshold_kw = float(peak_threshold_kw)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32) if continuous_action else spaces.Discrete(3)

        self.obs_columns = [
            "soc",
            "electricity_price_eur_per_kwh",
            "electricity_demand_kw",
            "renewable_generation_kw",
            "hour_of_day",
        ]
        if include_forecast:
            self.obs_columns += ["demand_forecast_1h_kw", "renewable_forecast_1h_kw"]

        self._price_max = max(0.5, float(self.data["electricity_price_eur_per_kwh"].quantile(0.995) * 1.5))
        self._demand_max = max(1.0, float(self.data["electricity_demand_kw"].quantile(0.995) * 1.3))
        self._renewable_max = max(1.0, float(self.data["renewable_generation_kw"].quantile(0.995) * 1.3))
        self._demand_fc_max = max(self._demand_max, float(self.data["demand_forecast_1h_kw"].max() * 1.1))
        self._renew_fc_max = max(self._renewable_max, float(self.data["renewable_forecast_1h_kw"].max() * 1.1))
        if normalize_observation:
            self.observation_space = spaces.Box(low=0.0, high=1.5, shape=(len(self.obs_columns),), dtype=np.float32)
        else:
            highs = [1.0, self._price_max, self._demand_max, self._renewable_max, 23.0]
            if include_forecast:
                highs += [self._demand_fc_max, self._renew_fc_max]
            self.observation_space = spaces.Box(low=np.zeros(len(highs), dtype=np.float32), high=np.array(highs, dtype=np.float32), dtype=np.float32)

        self.start_index = 0
        self.current_index = 0
        self.steps = 0
        self.soc = self.spec.initial_soc
        self.episode_log: list[dict[str, Any]] = []

    def _row(self) -> pd.Series:
        idx = min(self.current_index, len(self.data) - 1)
        return self.data.iloc[idx]

    def raw_state(self) -> np.ndarray:
        row = self._row()
        vals = [
            self.soc,
            float(row["electricity_price_eur_per_kwh"]),
            float(row["electricity_demand_kw"]),
            float(row["renewable_generation_kw"]),
            float(row["hour_of_day"]),
        ]
        if self.include_forecast:
            vals += [float(row["demand_forecast_1h_kw"]), float(row["renewable_forecast_1h_kw"])]
        return np.asarray(vals, dtype=np.float32)

    def _observation(self) -> np.ndarray:
        raw = self.raw_state().astype(np.float32)
        if not self.normalize_observation:
            return raw
        scale = np.asarray([1.0, self._price_max, self._demand_max, self._renewable_max, 23.0], dtype=np.float32)
        if self.include_forecast:
            scale = np.concatenate([scale, np.asarray([self._demand_fc_max, self._renew_fc_max], dtype=np.float32)])
        return np.clip(raw / scale, 0.0, 1.5).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        max_start = len(self.data) - self.episode_hours
        if "start_index" in options:
            self.start_index = int(np.clip(options["start_index"], 0, max_start))
        elif self.random_start and max_start > 0:
            self.start_index = int(self.np_random.integers(0, max_start + 1))
        else:
            self.start_index = 0
        self.current_index = self.start_index
        self.steps = 0
        self.soc = float(options.get("initial_soc", self.spec.initial_soc))
        self.soc = float(np.clip(self.soc, self.spec.min_soc, self.spec.max_soc))
        self.episode_log = []
        return self._observation(), {"raw_state": self.raw_state().copy(), "start_index": self.start_index}

    def valid_discrete_actions(self) -> list[int]:
        eps = 1e-8
        actions = [1]
        if self.soc < self.spec.max_soc - eps:
            actions.append(0)
        if self.soc > self.spec.min_soc + eps:
            actions.append(2)
        return sorted(actions)

    def _requested_power(self, action) -> float:
        if self.continuous_action:
            a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
            return a * self.spec.max_charge_kw if a >= 0 else a * self.spec.max_discharge_kw
        a = int(action)
        if a == 0:
            return self.spec.max_charge_kw
        if a == 1:
            return 0.0
        if a == 2:
            return -self.spec.max_discharge_kw
        raise ValueError(f"Invalid discrete action: {action}")

    def _clip_battery_power(self, requested_kw: float) -> tuple[float, float, float]:
        dt = self.spec.timestep_hours
        energy_kwh = self.soc * self.spec.capacity_kwh
        max_energy_kwh = self.spec.max_soc * self.spec.capacity_kwh
        min_energy_kwh = self.spec.min_soc * self.spec.capacity_kwh
        if requested_kw >= 0.0:
            soc_limited_bus_kw = max(0.0, (max_energy_kwh - energy_kwh) / (self.spec.charging_efficiency * dt))
            executed = min(requested_kw, self.spec.max_charge_kw, soc_limited_bus_kw)
            internal_delta_kwh = executed * self.spec.charging_efficiency * dt
            throughput_kwh = abs(internal_delta_kwh)
        else:
            requested_discharge = abs(requested_kw)
            soc_limited_bus_kw = max(0.0, (energy_kwh - min_energy_kwh) * self.spec.discharging_efficiency / dt)
            discharge_bus_kw = min(requested_discharge, self.spec.max_discharge_kw, soc_limited_bus_kw)
            executed = -discharge_bus_kw
            internal_delta_kwh = -discharge_bus_kw / self.spec.discharging_efficiency * dt
            throughput_kwh = abs(internal_delta_kwh)
        return float(executed), float(internal_delta_kwh), float(throughput_kwh)

    def step(self, action):
        row = self._row()
        requested_kw = self._requested_power(action)
        executed_kw, internal_delta_kwh, throughput_kwh = self._clip_battery_power(requested_kw)
        old_soc = self.soc
        new_energy = old_soc * self.spec.capacity_kwh + internal_delta_kwh
        self.soc = float(np.clip(new_energy / self.spec.capacity_kwh, self.spec.min_soc, self.spec.max_soc))

        charge_kw = max(executed_kw, 0.0)
        discharge_kw = max(-executed_kw, 0.0)
        demand_kw = float(row["electricity_demand_kw"])
        renewable_kw = float(row["renewable_generation_kw"])
        price = float(row["electricity_price_eur_per_kwh"])
        grid_power_kw = demand_kw - renewable_kw + charge_kw - discharge_kw
        grid_import_kw = max(grid_power_kw, 0.0)
        grid_export_kw = max(-grid_power_kw, 0.0)
        dt = self.spec.timestep_hours
        import_energy_kwh = grid_import_kw * dt
        export_energy_kwh = grid_export_kw * dt
        electricity_cost = import_energy_kwh * price - export_energy_kwh * self.spec.feed_in_tariff_eur_per_kwh
        requested_minus_executed_kwh = (requested_kw - executed_kw) * dt

        rb = calculate_reward(
            electricity_cost=electricity_cost,
            grid_import_kw=grid_import_kw,
            peak_threshold_kw=self.peak_threshold_kw,
            throughput_kwh=throughput_kwh,
            requested_minus_executed_kwh=requested_minus_executed_kwh,
            peak_penalty_eur_per_kw2=float(self.reward_config["peak_penalty_eur_per_kw2"]),
            degradation_eur_per_kwh_throughput=float(self.reward_config["degradation_eur_per_kwh_throughput"]),
            constraint_penalty_eur_per_kwh=float(self.reward_config["constraint_penalty_eur_per_kwh"]),
        )
        reward = float(self.reward_config.get("cost_weight", 1.0) * (-electricity_cost) - rb.peak_penalty - rb.degradation_penalty - rb.constraint_penalty)

        renewable_used_locally_kw = min(renewable_kw, demand_kw + charge_kw)
        constraint_violation = abs(requested_kw - executed_kw) > 1e-8
        log = {
            "timestamp": row.get("timestamp"),
            "step": self.steps,
            "soc_before": old_soc,
            "soc": self.soc,
            "price": price,
            "demand_kw": demand_kw,
            "renewable_kw": renewable_kw,
            "requested_battery_power_kw": requested_kw,
            "executed_battery_power_kw": executed_kw,
            "charge_kw": charge_kw,
            "discharge_kw": discharge_kw,
            "grid_power_kw": grid_power_kw,
            "grid_import_kw": grid_import_kw,
            "grid_export_kw": grid_export_kw,
            "electricity_cost": electricity_cost,
            "peak_penalty": rb.peak_penalty,
            "degradation_penalty": rb.degradation_penalty,
            "constraint_penalty": rb.constraint_penalty,
            "throughput_kwh": throughput_kwh,
            "renewable_used_locally_kwh": renewable_used_locally_kw * dt,
            "renewable_generation_kwh": renewable_kw * dt,
            "constraint_violation": int(constraint_violation),
            "reward": reward,
        }
        self.episode_log.append(log)

        self.steps += 1
        self.current_index += 1
        terminated = self.steps >= self.episode_hours
        truncated = self.current_index >= len(self.data)
        obs = self._observation() if not (terminated or truncated) else self._observation()
        info = log.copy()
        info["raw_state"] = self.raw_state().copy()
        return obs, reward, terminated, truncated, info

    def episode_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.episode_log)

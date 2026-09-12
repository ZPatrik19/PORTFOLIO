from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardBreakdown:
    electricity_cost: float
    peak_penalty: float
    degradation_penalty: float
    constraint_penalty: float

    @property
    def reward(self) -> float:
        return -(
            self.electricity_cost
            + self.peak_penalty
            + self.degradation_penalty
            + self.constraint_penalty
        )


def calculate_reward(
    *,
    electricity_cost: float,
    grid_import_kw: float,
    peak_threshold_kw: float,
    throughput_kwh: float,
    requested_minus_executed_kwh: float,
    peak_penalty_eur_per_kw2: float,
    degradation_eur_per_kwh_throughput: float,
    constraint_penalty_eur_per_kwh: float,
) -> RewardBreakdown:
    peak_excess = max(0.0, grid_import_kw - peak_threshold_kw)
    return RewardBreakdown(
        electricity_cost=float(electricity_cost),
        peak_penalty=float(peak_penalty_eur_per_kw2 * peak_excess**2),
        degradation_penalty=float(degradation_eur_per_kwh_throughput * throughput_kwh),
        constraint_penalty=float(constraint_penalty_eur_per_kwh * abs(requested_minus_executed_kwh)),
    )


def bad_reward_example(grid_import_kw: float, battery_power_kw: float) -> float:
    """Intentionally poor reward for documentation/teaching.

    It rewards discharge directly and can therefore incentivize wasteful cycling,
    regardless of electricity price or long-term battery value.
    """
    return -grid_import_kw + 0.5 * max(0.0, -battery_power_kw)

"""Optional bridge to CityLearn.

This project deliberately starts with `BatteryEnvironment` so the state/action/
transition/reward mechanics remain transparent. CityLearn is the next benchmark
layer, not the starting point.
"""
from __future__ import annotations


def check_citylearn_installation() -> str:
    try:
        import citylearn  # noqa: F401
        return "CityLearn is installed and can be used for the advanced benchmark."
    except ImportError:
        return (
            "CityLearn is optional. Install with `pip install citylearn` and then map "
            "the same evaluation KPI layer onto a CityLearn schema/building setup."
        )


def conceptual_mapping() -> dict[str, str]:
    return {
        "BatteryEnvironment.data": "CityLearn building/load/solar/pricing time series",
        "BatteryEnvironment.soc": "CityLearn electrical_storage state",
        "charge/idle/discharge": "CityLearn storage action",
        "electricity_cost reward": "CityLearn cost/reward functions",
        "single battery": "one or multiple buildings / district-level control",
    }

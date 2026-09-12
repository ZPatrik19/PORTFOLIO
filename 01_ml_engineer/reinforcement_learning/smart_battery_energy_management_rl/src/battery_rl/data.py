from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_energy_data(
    start: str = "2024-01-01",
    days: int = 365,
    seed: int = 42,
    add_noise: bool = True,
) -> pd.DataFrame:
    """Generate hourly, physically interpretable energy-management data.

    The synthetic generator intentionally separates *realized values* from
    *forecasts*. Forecasts are based on calendar-driven expectations plus their
    own noise and do not directly copy the next realized observation. This makes
    them reasonable stand-ins for information available at decision time.
    """
    rng = np.random.default_rng(seed)
    n = int(days * 24)
    ts = pd.date_range(start=start, periods=n, freq="h")
    hour = ts.hour.to_numpy()
    dow = ts.dayofweek.to_numpy()
    doy = ts.dayofyear.to_numpy()

    # Demand: base load + morning/evening peaks + weekday + seasonal effect.
    morning = 11.0 * np.exp(-0.5 * ((hour - 8.0) / 2.0) ** 2)
    evening = 24.0 * np.exp(-0.5 * ((hour - 19.0) / 2.8) ** 2)
    weekday = np.where(dow < 5, 7.0, -2.0)
    seasonal_demand = 6.0 * np.cos(2 * np.pi * (doy - 15) / 365.0)
    expected_demand = 42.0 + morning + evening + weekday + seasonal_demand
    demand_noise = rng.normal(0.0, 3.2, n) if add_noise else 0.0
    demand = np.clip(expected_demand + demand_noise, 15.0, None)

    # Solar-like renewable profile with seasonal daylight strength.
    solar_shape = np.maximum(0.0, np.sin(np.pi * (hour - 6.0) / 12.0))
    seasonal_solar = 0.55 + 0.45 * np.sin(2 * np.pi * (doy - 80) / 365.0)
    expected_renewable = 78.0 * solar_shape * np.clip(seasonal_solar, 0.12, 1.0)
    renewable_noise = rng.normal(0.0, 3.5, n) if add_noise else 0.0
    renewable = np.clip(expected_renewable + renewable_noise, 0.0, None)

    # Time-of-use price with scarcity component. Units are EUR/kWh.
    peak_price = 0.10 * np.exp(-0.5 * ((hour - 19.0) / 2.5) ** 2)
    midday_discount = -0.025 * np.exp(-0.5 * ((hour - 13.0) / 2.5) ** 2)
    scarcity = 0.0010 * np.maximum(demand - renewable - 45.0, 0.0)
    weekend_discount = np.where(dow >= 5, -0.012, 0.0)
    price_noise = rng.normal(0.0, 0.012, n) if add_noise else 0.0
    price = np.clip(0.105 + peak_price + midday_discount + scarcity + weekend_discount + price_noise, 0.025, 0.35)

    # Forecasts represent decision-time estimates, not actual future values.
    next_hour = (hour + 1) % 24
    next_doy = doy + (hour == 23)
    next_morning = 11.0 * np.exp(-0.5 * ((next_hour - 8.0) / 2.0) ** 2)
    next_evening = 24.0 * np.exp(-0.5 * ((next_hour - 19.0) / 2.8) ** 2)
    next_weekday = np.where(((dow + (hour == 23)) % 7) < 5, 7.0, -2.0)
    next_seasonal_demand = 6.0 * np.cos(2 * np.pi * (next_doy - 15) / 365.0)
    demand_forecast = 42.0 + next_morning + next_evening + next_weekday + next_seasonal_demand
    demand_forecast += rng.normal(0.0, 2.0, n)

    next_solar_shape = np.maximum(0.0, np.sin(np.pi * (next_hour - 6.0) / 12.0))
    next_seasonal_solar = 0.55 + 0.45 * np.sin(2 * np.pi * (next_doy - 80) / 365.0)
    renewable_forecast = 78.0 * next_solar_shape * np.clip(next_seasonal_solar, 0.12, 1.0)
    renewable_forecast += rng.normal(0.0, 2.8, n)
    renewable_forecast = np.clip(renewable_forecast, 0.0, None)

    df = pd.DataFrame({
        "timestamp": ts,
        "electricity_price_eur_per_kwh": price,
        "electricity_demand_kw": demand,
        "renewable_generation_kw": renewable,
        "hour_of_day": hour,
        "day_of_week": dow,
        "demand_forecast_1h_kw": np.clip(demand_forecast, 10.0, None),
        "renewable_forecast_1h_kw": renewable_forecast,
    })
    df["rolling_average_price_24h"] = df["electricity_price_eur_per_kwh"].rolling(24, min_periods=1).mean()
    return df


def download_energy_data(url: str, destination: str | Path, *, timeout: int = 60) -> Path:
    """Download a CSV dataset to ``destination`` using only the standard library.

    This helper is intentionally small: the default project dataset is synthetic
    and needs no network access, but a real CSV can be configured later through
    ``config.yaml`` with ``data.source: url``.
    """
    from pathlib import Path
    from urllib.request import Request, urlopen

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "smart-battery-energy-rl/1.0"})
    with urlopen(request, timeout=timeout) as response, destination.open("wb") as fh:
        fh.write(response.read())
    return destination


def validate_energy_dataframe(df: pd.DataFrame) -> None:
    """Validate the minimum schema required by the battery environment."""
    required = {
        "timestamp",
        "electricity_price_eur_per_kwh",
        "electricity_demand_kw",
        "renewable_generation_kw",
        "hour_of_day",
        "day_of_week",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Energy dataset is missing required columns: {missing}")
    if len(df) < 48:
        raise ValueError("Energy dataset must contain at least 48 hourly rows.")


def temporal_split(df: pd.DataFrame, train_fraction: float = 0.70, validation_fraction: float = 0.15):
    """Chronological split to avoid training on future periods."""
    n = len(df)
    n_train = int(n * train_fraction)
    n_val = int(n * validation_fraction)
    train = df.iloc[:n_train].reset_index(drop=True)
    val = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test = df.iloc[n_train + n_val:].reset_index(drop=True)
    return train, val, test


def scenario_transform(df: pd.DataFrame, scenario: str, seed: int = 0) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    p = "electricity_price_eur_per_kwh"
    r = "renewable_generation_kw"
    if scenario == "normal":
        return out
    if scenario == "high_price":
        out[p] = np.clip(out[p] * 1.65, 0.0, None)
    elif scenario == "volatile_price":
        shocks = rng.normal(0.0, 0.055, len(out))
        spikes = (rng.random(len(out)) < 0.06) * rng.uniform(0.05, 0.18, len(out))
        out[p] = np.clip(out[p] + shocks + spikes, 0.01, 0.55)
    elif scenario == "high_renewable":
        out[r] = out[r] * 1.65
    elif scenario == "low_renewable":
        out[r] = out[r] * 0.35
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return out

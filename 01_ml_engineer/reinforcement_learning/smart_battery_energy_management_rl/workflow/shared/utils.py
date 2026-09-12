from __future__ import annotations
from pathlib import Path
import json, sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from battery_rl.config import load_config
from battery_rl.data import (
    download_energy_data,
    generate_synthetic_energy_data,
    temporal_split,
    validate_energy_dataframe,
)


def ensure_dirs() -> None:
    for rel in [
        "01_data/raw", "01_data/processed", "04_results/figures", "04_results/metrics",
        "04_results/models", "04_results/predictions", "05_graphviz"
    ]:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)


def _read_energy_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if "hour_of_day" not in df.columns:
        df["hour_of_day"] = df["timestamp"].dt.hour
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["timestamp"].dt.dayofweek
    if "rolling_average_price_24h" not in df.columns and "electricity_price_eur_per_kwh" in df.columns:
        df["rolling_average_price_24h"] = (
            df["electricity_price_eur_per_kwh"].rolling(24, min_periods=1).mean()
        )
    validate_energy_dataframe(df)
    return df


def ensure_data(config_path: str | Path | None = None, *, force: bool = False):
    """Ensure raw and chronological train/validation/test data exist.

    ``data.source`` controls acquisition:
    - ``synthetic`` (default): generate the deterministic educational dataset locally;
    - ``url``: download a CSV from ``data.url`` when the raw file is absent.

    No download is attempted for the default synthetic configuration.
    """
    ensure_dirs()
    cfg = load_config(config_path)
    dcfg = cfg["data"]
    source = str(dcfg.get("source", "synthetic")).lower().strip()
    raw_filename = str(dcfg.get("raw_filename", "synthetic_energy_timeseries.csv"))

    raw_path = ROOT / "01_data/raw" / raw_filename
    full_path = ROOT / "01_data/processed/energy_timeseries.csv"
    train_path = ROOT / "01_data/processed/train.csv"
    val_path = ROOT / "01_data/processed/validation.csv"
    test_path = ROOT / "01_data/processed/test.csv"
    outputs = [raw_path, full_path, train_path, val_path, test_path]

    if force:
        for path in outputs:
            if path.exists():
                path.unlink()

    if not all(path.exists() for path in outputs):
        if source == "synthetic":
            df = generate_synthetic_energy_data(
                start=dcfg["start"],
                days=int(dcfg["days"]),
                seed=int(cfg["project"]["seed"]),
                add_noise=bool(dcfg.get("synthetic_noise", True)),
            )
            df.to_csv(raw_path, index=False)
        elif source == "url":
            url = dcfg.get("url")
            if not url:
                raise ValueError("config.yaml requires data.url when data.source is 'url'.")
            if not raw_path.exists():
                print(f"Downloading dataset from: {url}")
                download_energy_data(str(url), raw_path)
            df = _read_energy_csv(raw_path)
        else:
            raise ValueError(f"Unsupported data.source: {source!r}. Use 'synthetic' or 'url'.")

        validate_energy_dataframe(df)
        train, val, test = temporal_split(
            df,
            float(dcfg["train_fraction"]),
            float(dcfg["validation_fraction"]),
        )
        df.to_csv(full_path, index=False)
        train.to_csv(train_path, index=False)
        val.to_csv(val_path, index=False)
        test.to_csv(test_path, index=False)

    return tuple(_read_energy_csv(path) for path in [full_path, train_path, val_path, test_path])


def build_env(df, cfg, *, continuous=False, include_forecast=None, reward_override=None, random_start=True):
    from battery_rl.environment import BatteryEnvironment
    ecfg = cfg["environment"]
    reward_cfg = dict(cfg["reward"])
    if reward_override:
        reward_cfg.update(reward_override)
    return BatteryEnvironment(
        df,
        cfg["battery"],
        reward_cfg,
        episode_hours=int(ecfg["episode_hours"]),
        random_start=random_start,
        continuous_action=continuous,
        normalize_observation=bool(ecfg.get("normalize_observation", True)),
        include_forecast=bool(ecfg.get("include_forecast", False) if include_forecast is None else include_forecast),
        peak_threshold_kw=float(ecfg["peak_threshold_kw"]),
    )


def save_json(obj, relative_path: str) -> Path:
    p = ROOT / relative_path
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)
    return p


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

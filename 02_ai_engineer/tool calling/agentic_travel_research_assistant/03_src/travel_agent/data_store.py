from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any


def project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "01_data").exists() and (parent / "pyproject.toml").exists():
            return parent
    return here.parents[2]


def raw_data_dir() -> Path:
    return project_root() / "01_data" / "raw"


@lru_cache(maxsize=None)
def read_csv(name: str) -> tuple[dict[str, str], ...]:
    path = raw_data_dir() / name
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return tuple(csv.DictReader(f))


def _norm(value: str) -> str:
    return value.strip().casefold()


def city_record(city: str) -> dict[str, str] | None:
    key = _norm(city)
    aliases = {
        "bécs": "vienna", "becs": "vienna", "prága": "prague", "praga": "prague",
        "párizs": "paris", "parizs": "paris", "róma": "rome", "roma": "rome",
        "münchen": "munich", "munchen": "munich", "krakkó": "krakow", "krakko": "krakow",
    }
    key = aliases.get(key, key)
    for row in read_csv("cities.csv"):
        if _norm(row["city"]) == key:
            return dict(row)
    return None


@lru_cache(maxsize=None)
def _rows_for_city_cached(filename: str, canonical: str) -> tuple[dict[str, str], ...]:
    key = _norm(canonical)
    return tuple(dict(row) for row in read_csv(filename) if _norm(row.get("city", "")) == key)


def rows_for_city(filename: str, city: str) -> list[dict[str, str]]:
    rec = city_record(city)
    canonical = rec["city"] if rec else city
    # Return shallow copies so callers cannot mutate the cached catalogue.
    return [dict(row) for row in _rows_for_city_cached(filename, canonical)]


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default

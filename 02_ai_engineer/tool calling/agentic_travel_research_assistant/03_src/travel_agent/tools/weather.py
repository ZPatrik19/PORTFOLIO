from __future__ import annotations

import json
import os
from datetime import date
from typing import Literal
from urllib.parse import urlencode
from urllib.request import urlopen

from pydantic import BaseModel, ConfigDict, Field

from travel_agent.data_store import as_float, city_record, rows_for_city
from .base import BaseTool


class WeatherInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str = Field(min_length=2)
    days: int = Field(ge=1, le=16)
    unit: Literal["celsius", "fahrenheit"] = "celsius"


class WeatherTool(BaseTool[WeatherInput]):
    name = "get_weather"
    description = "Get a multi-day weather forecast. It tries the live Open-Meteo API first and automatically falls back to the local deterministic weather dataset when network access is unavailable."
    input_model = WeatherInput

    @staticmethod
    def _live(city: str, days: int) -> dict:
        rec = city_record(city)
        if not rec:
            raise ValueError(f"Unknown city: {city}")
        params = urlencode({
            "latitude": rec["latitude"], "longitude": rec["longitude"],
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
            "timezone": "auto", "forecast_days": days,
        })
        with urlopen(f"https://api.open-meteo.com/v1/forecast?{params}", timeout=4) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        daily = payload["daily"]
        forecast=[]
        for i,d in enumerate(daily["time"]):
            forecast.append({
                "date": d, "temp_min_c": daily["temperature_2m_min"][i], "temp_max_c": daily["temperature_2m_max"][i],
                "precipitation_mm": daily["precipitation_sum"][i], "weather_code": daily["weather_code"][i],
                "wind_kph": daily["wind_speed_10m_max"][i],
            })
        return {"city": rec["city"], "forecast": forecast, "source": "Open-Meteo live API"}

    @staticmethod
    def _fallback(city: str, days: int) -> dict:
        rows = rows_for_city("weather_fallback.csv", city)
        if not rows:
            return {"error": f"No local weather fallback for city: {city}"}
        out=[]
        for r in rows[:days]:
            out.append({
                "date": r["date"], "temp_min_c": as_float(r["temp_min_c"]), "temp_max_c": as_float(r["temp_max_c"]),
                "precipitation_mm": as_float(r["precipitation_mm"]), "condition": r["condition"], "wind_kph": as_float(r["wind_kph"]),
            })
        return {"city": city, "forecast": out, "source": "01_data/raw/weather_fallback.csv", "fallback": True}

    def execute(self, arguments: WeatherInput) -> dict:
        mode = os.getenv("TRAVEL_DATA_MODE", "auto").strip().lower()
        if mode == "local":
            result = self._fallback(arguments.city, arguments.days)
        elif mode == "live":
            result = self._live(arguments.city, arguments.days)
        else:
            try:
                result = self._live(arguments.city, arguments.days)
            except Exception as exc:
                result = self._fallback(arguments.city, arguments.days)
                result["live_api_error"] = type(exc).__name__
        if arguments.unit == "fahrenheit" and "forecast" in result:
            for row in result["forecast"]:
                if "temp_min_c" in row:
                    row["temp_min_f"] = round(row["temp_min_c"] * 9 / 5 + 32, 1)
                    row["temp_max_f"] = round(row["temp_max_c"] * 9 / 5 + 32, 1)
        return result

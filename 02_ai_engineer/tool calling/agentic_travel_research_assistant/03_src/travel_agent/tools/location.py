from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from travel_agent.data_store import as_float, city_record
from .base import BaseTool


class LocationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str = Field(min_length=2)


class LocationTool(BaseTool[LocationInput]):
    name = "get_location_info"
    description = "Look up destination metadata from the local city catalogue: country, currency, coordinates, timezone, language and indicative local daily costs."
    input_model = LocationInput

    def execute(self, arguments: LocationInput) -> dict:
        row = city_record(arguments.city)
        if not row:
            return {"error": f"Unknown city in local catalogue: {arguments.city}"}
        return {
            "city": row["city"], "country": row["country"], "country_code": row["country_code"],
            "currency": row["currency"], "language": row["language"], "timezone": row["timezone"],
            "latitude": as_float(row["latitude"]), "longitude": as_float(row["longitude"]),
            "daily_food_budget_eur": as_float(row["daily_food_budget_eur"]),
            "daily_transport_eur": as_float(row["daily_transport_eur"]),
            "tourism_score": as_float(row["tourism_score"]),
            "source": "01_data/raw/cities.csv",
        }

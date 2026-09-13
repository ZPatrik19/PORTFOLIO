from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from travel_agent.data_store import as_float, as_int, rows_for_city
from .base import BaseTool


class HotelSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str = Field(min_length=2)
    nights: int = Field(ge=1, le=30)
    max_price_per_night_eur: float | None = Field(default=None, ge=1)
    min_rating: float = Field(default=0.0, ge=0.0, le=5.0)
    top_k: int = Field(default=5, ge=1, le=20)


class HotelSearchTool(BaseTool[HotelSearchInput]):
    name = "search_hotels"
    description = "Search and rank a 180,000-row local hotel inventory by city, price and rating. Data is synthetic and deterministic, intended for agent/tool evaluation rather than booking."
    input_model = HotelSearchInput

    def execute(self, arguments: HotelSearchInput) -> dict:
        rows = rows_for_city("hotels.csv", arguments.city)
        if not rows:
            return {"error": f"No hotels found for city: {arguments.city}"}
        parsed = []
        for r in rows:
            nightly = as_float(r["nightly_eur"])
            rating = as_float(r["rating"])
            if arguments.max_price_per_night_eur is not None and nightly > arguments.max_price_per_night_eur:
                continue
            if rating < arguments.min_rating:
                continue
            parsed.append({
                "hotel_id": r["hotel_id"], "name": r["name"], "area": r["area"],
                "stars": as_int(r["stars"]), "rating": rating, "nightly_eur": nightly,
                "review_count": as_int(r["review_count"]), "breakfast_included": bool(as_int(r["breakfast_included"])),
                "refundable": bool(as_int(r["refundable"])), "distance_to_center_km": as_float(r["distance_to_center_km"]),
                "total_eur": round(nightly * arguments.nights, 2),
            })
        parsed.sort(key=lambda x: (-x["rating"], x["nightly_eur"], -x["review_count"]))
        return {
            "city": arguments.city, "nights": arguments.nights, "matched": len(parsed),
            "results": parsed[: arguments.top_k], "source": "01_data/raw/hotels.csv",
            "data_note": "Synthetic inventory; search/filter/ranking are real, availability is not live.",
        }

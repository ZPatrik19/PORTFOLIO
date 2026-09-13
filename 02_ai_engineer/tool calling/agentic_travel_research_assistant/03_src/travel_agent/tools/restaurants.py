from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from travel_agent.data_store import as_float, as_int, rows_for_city
from .base import BaseTool


class RestaurantSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str = Field(min_length=2)
    cuisines: list[str] | None = None
    max_meal_eur: float | None = Field(default=None, ge=1)
    min_rating: float = Field(default=0.0, ge=0.0, le=5.0)
    vegetarian_only: bool = False
    top_k: int = Field(default=6, ge=1, le=20)


class RestaurantSearchTool(BaseTool[RestaurantSearchInput]):
    name = "search_restaurants"
    description = "Search and rank a 90,000-row synthetic restaurant catalogue by city, cuisine, meal budget, rating and vegetarian availability."
    input_model = RestaurantSearchInput

    def execute(self, arguments: RestaurantSearchInput) -> dict:
        rows = rows_for_city("restaurants.csv", arguments.city)
        if not rows:
            return {"error": f"No restaurants found for city: {arguments.city}"}
        wanted = {x.casefold() for x in arguments.cuisines} if arguments.cuisines else None
        parsed = []
        for r in rows:
            meal = as_float(r["avg_meal_eur"])
            rating = as_float(r["rating"])
            if wanted and r["cuisine"].casefold() not in wanted:
                continue
            if arguments.max_meal_eur is not None and meal > arguments.max_meal_eur:
                continue
            if rating < arguments.min_rating:
                continue
            if arguments.vegetarian_only and not bool(as_int(r["vegetarian_options"])):
                continue
            parsed.append({
                "restaurant_id": r["restaurant_id"], "name": r["name"], "cuisine": r["cuisine"],
                "rating": rating, "price_level": as_int(r["price_level"]), "avg_meal_eur": meal,
                "vegetarian_options": bool(as_int(r["vegetarian_options"])),
                "vegan_options": bool(as_int(r["vegan_options"])),
                "reservation_recommended": bool(as_int(r["reservation_recommended"])),
                "family_friendly": bool(as_int(r["family_friendly"])), "area": r["area"],
            })
        parsed.sort(key=lambda x: (-x["rating"], x["avg_meal_eur"]))
        return {
            "city": arguments.city, "matched": len(parsed), "results": parsed[:arguments.top_k],
            "source": "01_data/raw/restaurants.csv",
            "data_note": "Synthetic restaurant catalogue for tool-routing and filtering experiments.",
        }

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from travel_agent.data_store import as_float, as_int, rows_for_city
from .base import BaseTool


class AttractionSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str = Field(min_length=2)
    categories: list[str] | None = None
    max_ticket_eur: float | None = Field(default=None, ge=0)
    top_k: int = Field(default=6, ge=1, le=20)


class AttractionSearchTool(BaseTool[AttractionSearchInput]):
    name = "search_attractions"
    description = "Search a 90,000-row attraction catalogue by city, optional categories and ticket-price limit. Useful for itinerary research."
    input_model = AttractionSearchInput

    def execute(self, arguments: AttractionSearchInput) -> dict:
        rows = rows_for_city("attractions.csv", arguments.city)
        wanted = {x.casefold() for x in arguments.categories} if arguments.categories else None
        out = []
        for r in rows:
            ticket = as_float(r["ticket_eur"])
            if wanted and r["category"].casefold() not in wanted:
                continue
            if arguments.max_ticket_eur is not None and ticket > arguments.max_ticket_eur:
                continue
            out.append({
                "poi_id": r["poi_id"], "name": r["name"], "category": r["category"],
                "rating": as_float(r["rating"]), "ticket_eur": ticket,
                "estimated_visit_hours": as_float(r["estimated_visit_hours"]),
                "indoor": bool(as_int(r["indoor"])), "family_friendly": bool(as_int(r["family_friendly"])),
            })
        out.sort(key=lambda x: (-x["rating"], x["ticket_eur"]))
        return {"city": arguments.city, "matched": len(out), "results": out[:arguments.top_k], "source": "01_data/raw/attractions.csv"}

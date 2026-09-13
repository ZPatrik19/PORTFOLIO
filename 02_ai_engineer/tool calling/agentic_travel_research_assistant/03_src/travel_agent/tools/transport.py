from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from travel_agent.data_store import as_float, rows_for_city
from .base import BaseTool


class TransportInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str = Field(min_length=2)
    days: int = Field(ge=1, le=30)


class TransportTool(BaseTool[TransportInput]):
    name = "get_transport_options"
    description = "Return local public-transport ticket/pass prices and estimate the best simple pass cost for a trip length."
    input_model = TransportInput

    def execute(self, arguments: TransportInput) -> dict:
        rows = rows_for_city("transport.csv", arguments.city)
        if not rows:
            return {"error": f"No transport profile for city: {arguments.city}"}
        r = rows[0]
        day = as_float(r["day_pass_eur"])
        three = as_float(r["three_day_pass_eur"])
        if arguments.days == 1:
            estimated = day
            product = "day pass"
        else:
            three_count, remainder = divmod(arguments.days, 3)
            option_combo = three_count * three + remainder * day
            option_daily = arguments.days * day
            estimated = min(option_combo, option_daily)
            product = "3-day/day-pass combination" if option_combo <= option_daily else "daily passes"
        return {
            "city": arguments.city, "days": arguments.days,
            "single_ticket_eur": as_float(r["single_ticket_eur"]), "day_pass_eur": day,
            "three_day_pass_eur": three, "airport_transfer_eur": as_float(r["airport_transfer_eur"]),
            "bike_day_eur": as_float(r["bike_day_eur"]), "walkability_score": as_float(r["walkability_score"]),
            "estimated_pass_cost_eur": round(estimated, 2), "recommended_product": product,
            "source": "01_data/raw/transport.csv",
        }

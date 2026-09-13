from travel_agent.tools import ToolRegistry
from travel_agent.tools.currency import CurrencyTool
from travel_agent.tools.weather import WeatherTool

def test_currency_conversion_fallback(monkeypatch):
    def fail(*args,**kwargs):raise OSError("offline")
    monkeypatch.setattr(CurrencyTool,"_live",staticmethod(fail))
    out,_,success,_=ToolRegistry().execute("convert_currency",{"amount":500,"from_currency":"EUR","to_currency":"HUF"})
    assert success and out["converted_amount"]==197500.0 and out["fallback"] is True

def test_hotel_inventory_filter():
    out,_,success,_=ToolRegistry().execute("search_hotels",{"city":"Prague","nights":2,"max_price_per_night_eur":150,"min_rating":4.0,"top_k":5})
    assert success and len(out["results"])<=5
    assert all(h["nightly_eur"]<=150 and h["rating"]>=4.0 for h in out["results"])

def test_attraction_filter():
    out,_,success,_=ToolRegistry().execute("search_attractions",{"city":"Vienna","categories":["museum","landmark"],"max_ticket_eur":20,"top_k":6})
    assert success and out["results"]
    assert all(x["category"] in {"museum","landmark"} and x["ticket_eur"]<=20 for x in out["results"])

def test_transport_cost():
    out,_,success,_=ToolRegistry().execute("get_transport_options",{"city":"Vienna","days":3})
    assert success and out["estimated_pass_cost_eur"]>0

def test_weather_fallback(monkeypatch):
    def fail(*args,**kwargs):raise OSError("offline")
    monkeypatch.setattr(WeatherTool,"_live",staticmethod(fail))
    out,_,success,_=ToolRegistry().execute("get_weather",{"city":"Vienna","days":3,"unit":"celsius"})
    assert success and len(out["forecast"])==3 and out["fallback"] is True


def test_restaurant_search_tool_returns_filtered_results():
    from travel_agent.tools.registry import ToolRegistry
    registry = ToolRegistry()
    out, _, ok, err = registry.execute("search_restaurants", {
        "city": "Vienna", "cuisines": None, "max_meal_eur": 35,
        "min_rating": 4.0, "vegetarian_only": False, "top_k": 6,
    })
    assert ok, err
    assert out["matched"] >= 1
    assert len(out["results"]) <= 6
    assert all(r["avg_meal_eur"] <= 35 for r in out["results"])
    assert all(r["rating"] >= 4.0 for r in out["results"])

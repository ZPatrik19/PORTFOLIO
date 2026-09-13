from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from travel_agent.i18n import normalize_language, t
from travel_agent.models import AgentRun, ToolCallRecord
from travel_agent.tools import ToolRegistry
from .heuristics import attraction_constraints, currency_request_enhanced, detect_intents, extract_city, extract_days_enhanced, hotel_constraints, restaurant_constraints


@dataclass
class PlanStep:
    step:int; tool:str; reason:str; arguments:dict[str,Any]; depends_on:list[str]


class PlanThenExecuteTravelAgent:
    """Inspectable deterministic planner/executor used as an offline benchmark."""
    def __init__(self, registry:ToolRegistry|None=None, language:str="en"):
        self.registry=registry or ToolRegistry();self.language=normalize_language(language)

    def create_plan(self,query:str)->list[PlanStep]:
        city=extract_city(query);days=extract_days_enhanced(query);intents=detect_intents(query);fx=currency_request_enhanced(query);steps=[]
        def add(tool,reason,args,deps=None):steps.append(PlanStep(len(steps)+1,tool,reason,args,deps or []))
        if city and (intents["location"] or intents["budget"]): add("get_location_info","Destination metadata/cost profile required.",{"city":city})
        if city and intents["weather"]: add("get_weather","Weather requested.",{"city":city,"days":days,"unit":"celsius"})
        if fx:
            amount,src,dst=fx;add("convert_currency","Currency conversion requested.",{"amount":amount,"from_currency":src,"to_currency":dst})
        if city and intents["hotel"]:
            price,rating,top_k=hotel_constraints(query);add("search_hotels","Accommodation search requested.",{"city":city,"nights":days,"max_price_per_night_eur":price,"min_rating":rating,"top_k":top_k})
        if city and intents["attractions"]:
            cats,max_ticket,top_k=attraction_constraints(query);add("search_attractions","Attraction research requested.",{"city":city,"categories":cats,"max_ticket_eur":max_ticket,"top_k":top_k})
        if city and intents["restaurant"]:
            cuisines,max_meal,min_rating,vegetarian,top_k=restaurant_constraints(query);add("search_restaurants","Restaurant search requested.",{"city":city,"cuisines":cuisines,"max_meal_eur":max_meal,"min_rating":min_rating,"vegetarian_only":vegetarian,"top_k":top_k})
        if city and intents["transport"]: add("get_transport_options","Local transport requested.",{"city":city,"days":days})
        if city and intents["budget"]:
            if not any(s.tool=="get_location_info" for s in steps):add("get_location_info","Budget needs daily cost profile.",{"city":city})
            if not intents["exclude_hotel"] and not any(s.tool=="search_hotels" for s in steps):add("search_hotels","Budget includes accommodation.",{"city":city,"nights":days,"max_price_per_night_eur":None,"min_rating":0.0,"top_k":5})
            if not any(s.tool=="get_transport_options" for s in steps):add("get_transport_options","Budget includes transport.",{"city":city,"days":days})
            add("calculate","Combine tool outputs into a local budget.",{"expression":"<resolved-after-dependencies>"},["get_location_info","search_hotels","get_transport_options"])
        return steps

    def run(self,query:str)->AgentRun:
        started=time.perf_counter();plan=self.create_plan(query);trace=[];outputs={};days=extract_days_enhanced(query)
        for s in plan:
            args=dict(s.arguments)
            if s.tool=="calculate":
                loc=outputs.get("get_location_info",{});hot=outputs.get("search_hotels",{});trans=outputs.get("get_transport_options",{})
                hotel=min((float(h["total_eur"]) for h in hot.get("results",[])),default=0.0)
                food=float(loc.get("daily_food_budget_eur",0))*days
                transit=float(trans.get("estimated_pass_cost_eur",0))
                args={"expression":f"{hotel:g} + {food:g} + {transit:g}"}
            out,lat,ok,err=self.registry.execute(s.tool,args);trace.append(ToolCallRecord(s.step,s.tool,args,out,lat,ok,err));outputs[s.tool]=out
        lines=[]
        hu = self.language == "hu"
        if "get_location_info" in outputs and "error" not in outputs["get_location_info"]:
            x=outputs["get_location_info"]
            text = f"{x['city']}, {x['country']} ({x['currency']})."
            lines.append(f"{t('destination',self.language)}: {text}")
        if "get_weather" in outputs and "error" not in outputs["get_weather"]:
            x=outputs["get_weather"]
            preview=[]
            for d in x["forecast"][:5]:
                lo=d.get("temp_min_c"); hi=d.get("temp_max_c"); rain=d.get("precipitation_mm",0)
                preview.append(f"{d['date']}: {lo:g}–{hi:g}°C, " + (f"csapadék {rain:g} mm" if hu else f"rain {rain:g} mm"))
            lines.append(f"{t('weather',self.language)} ({x['source']}): " + "; ".join(preview))
        if "convert_currency" in outputs and "error" not in outputs["convert_currency"]:
            x=outputs["convert_currency"]
            lines.append(f"{t('currency',self.language)}: {x['amount']:g} {x['from_currency']} ≈ {x['converted_amount']:g} {x['to_currency']} ({x['source']}).")
        if "search_hotels" in outputs and "error" not in outputs["search_hotels"]:
            x=outputs["search_hotels"]
            opts=", ".join(f"{h['name']} (€{h['nightly_eur']:.0f}/" + ("éj" if hu else "night") + f", {h['rating']:.1f}/5)" for h in x["results"][:3])
            prefix = f"{x['matched']} találat. Legjobb opciók" if hu else f"{x['matched']} matches. Top options"
            lines.append(f"{t('hotel',self.language)}: {prefix}: {opts}.")
        if "search_attractions" in outputs and "error" not in outputs["search_attractions"]:
            x=outputs["search_attractions"]
            opts=", ".join(f"{a['name']} ({a['category']}, €{a['ticket_eur']:.0f})" for a in x["results"][:4])
            lines.append(f"{t('attractions',self.language)}: {opts}.")
        if "search_restaurants" in outputs and "error" not in outputs["search_restaurants"]:
            x=outputs["search_restaurants"]
            opts=", ".join(f"{r['name']} ({r['cuisine']}, €{r['avg_meal_eur']:.0f}, {r['rating']:.1f}/5)" for r in x["results"][:4])
            lines.append(("Éttermek" if hu else "Restaurants") + f": {opts}.")
        if "get_transport_options" in outputs and "error" not in outputs["get_transport_options"]:
            x=outputs["get_transport_options"]
            if hu:
                txt=f"napijegy €{x['day_pass_eur']:.2f}, 3 napos bérlet €{x['three_day_pass_eur']:.2f}; becsült {x['days']} napos költség €{x['estimated_pass_cost_eur']:.2f}."
            else:
                txt=f"day pass €{x['day_pass_eur']:.2f}, 3-day pass €{x['three_day_pass_eur']:.2f}; estimated {x['days']}-day pass cost €{x['estimated_pass_cost_eur']:.2f}."
            lines.append(f"{t('transport',self.language)}: {txt}")
        if "calculate" in outputs and "error" not in outputs["calculate"]:
            label=t('budget',self.language); suffix="Becsült helyi költség" if hu else "Local budget estimate"
            lines.append(f"{label}: {suffix}: €{outputs['calculate']['result']:.2f}.")
        answer="\n".join(lines) if lines else "No supported travel tool was required."
        return AgentRun(query=query,answer=answer,trace=trace,total_latency_ms=(time.perf_counter()-started)*1000,model="offline-plan-then-execute",metadata={"methodology":"plan_execute","plan":[asdict(x) for x in plan]})

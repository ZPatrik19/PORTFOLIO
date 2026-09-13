from __future__ import annotations

from travel_agent.config import Settings
from .offline_agent import OfflineTravelAgent
from .openai_agent import OpenAITravelAgent
from .ml_router_agent import MLRouterTravelAgent
from .plan_execute_agent import PlanThenExecuteTravelAgent


def resolved_methodology(settings: Settings) -> str:
    if settings.methodology != "auto":
        return settings.methodology
    return "openai_direct" if settings.mode == "openai" else "rule_based"


def build_agent(settings: Settings):
    methodology = resolved_methodology(settings)

    if methodology == "rule_based":
        return OfflineTravelAgent(language=settings.language)
    if methodology == "plan_execute":
        return PlanThenExecuteTravelAgent(language=settings.language)
    if methodology == "ml_router":
        return MLRouterTravelAgent(language=settings.language)
    if methodology == "openai_direct":
        if not settings.openai_api_key:
            raise RuntimeError("openai_direct requires OPENAI_API_KEY. Copy .env.example to .env and add your key.")
        return OpenAITravelAgent(
            api_key=settings.openai_api_key,
            model=settings.model,
            max_steps=settings.max_agent_steps,
            language=settings.language,
        )
    raise ValueError("AGENT_METHODOLOGY must be one of: auto, rule_based, plan_execute, ml_router, openai_direct.")

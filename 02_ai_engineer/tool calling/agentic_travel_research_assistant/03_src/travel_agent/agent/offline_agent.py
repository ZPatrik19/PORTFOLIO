from __future__ import annotations

from .plan_execute_agent import PlanThenExecuteTravelAgent, PlanStep


class OfflineTravelAgent(PlanThenExecuteTravelAgent):
    """Small deterministic baseline used for methodology comparison.

    It deliberately supports only the original core capabilities (location, weather,
    currency and hotels). The richer plan_execute strategy additionally routes
    attraction and transport requests, so the benchmark can measure a real difference.
    """

    def create_plan(self, query: str) -> list[PlanStep]:
        plan = super().create_plan(query)
        unsupported = {"search_attractions", "get_transport_options", "search_restaurants"}
        return [step for step in plan if step.tool not in unsupported]

    def run(self, query: str):
        run = super().run(query)
        run.model = "offline-rule-based-baseline"
        run.metadata["methodology"] = "rule_based"
        return run

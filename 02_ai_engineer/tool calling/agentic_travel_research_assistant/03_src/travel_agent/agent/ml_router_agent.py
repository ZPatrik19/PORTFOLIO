from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from .heuristics import attraction_constraints, currency_request_enhanced, detect_intents, extract_city, extract_days_enhanced, hotel_constraints, restaurant_constraints
from .plan_execute_agent import PlanStep, PlanThenExecuteTravelAgent


class MLRouterTravelAgent(PlanThenExecuteTravelAgent):
    """Word TF-IDF multi-label router backed by the 240k bilingual query corpus.

    The classifier chooses capabilities. Deterministic parsers still extract structured
    arguments, which keeps the experiment inspectable and separates routing quality
    from argument extraction quality.
    """

    def __init__(self, registry=None, language: str = "en", model_path: str | Path | None = None):
        super().__init__(registry=registry, language=language)
        from travel_agent.training import load_router_artifact
        root = Path(__file__).resolve().parents[3]
        default_path = root / "06_results" / "models" / "intent_router.joblib"
        model_path = Path(model_path) if model_path else default_path
        if model_path != default_path:
            # Custom artifacts are an advanced escape hatch; the project setup and
            # compatibility sidecar govern the default model path.
            import joblib
            artifact = joblib.load(model_path)
        else:
            artifact = load_router_artifact()
        self.pipeline = artifact["pipeline"]
        self.mlb = artifact["mlb"]
        self.threshold = float(artifact.get("threshold", 0.45))
        self.thresholds = artifact.get("thresholds", {})
        self.router_metrics = artifact.get("metrics", {})

    def predict_intents(self, query: str) -> tuple[set[str], dict[str, float]]:
        # Long multi-intent requests dilute TF-IDF probabilities. Score the full query
        # plus short clauses and retain the maximum probability per label.
        clauses = [query]
        clauses.extend(x.strip() for x in re.split(r"[.!?;]+|\b(?:and|then|plus|also|és|majd|valamint)\b", query, flags=re.I) if len(x.strip()) >= 4)
        prob_matrix = self.pipeline.predict_proba(clauses)
        max_probs = prob_matrix.max(axis=0)
        scores = {str(label): float(prob) for label, prob in zip(self.mlb.classes_, max_probs)}
        chosen = {label for label, prob in scores.items() if prob >= float(self.thresholds.get(label, self.threshold))}

        # The supervised corpus contains indirect phrases as well. For a clause whose
        # top class is clearly separated, preserve that local intent even when its
        # absolute probability is modest. This is useful for composed requests.
        for row in prob_matrix[1:]:
            order = row.argsort()[::-1]
            top, second = int(order[0]), int(order[1])
            top_p, second_p = float(row[top]), float(row[second])
            if top_p >= 0.12 and (top_p >= self.threshold or top_p >= second_p * 1.20):
                chosen.add(str(self.mlb.classes_[top]))
            if second_p >= 0.65:
                chosen.add(str(self.mlb.classes_[second]))
        if not chosen and scores:
            chosen = {max(scores, key=scores.get)}
        return chosen, scores

    def create_plan(self, query: str) -> list[PlanStep]:
        city = extract_city(query)
        days = extract_days_enhanced(query)
        predicted, scores = self.predict_intents(query)
        # High-precision lexical cues act as a guardrail for explicit requests; the
        # classifier remains useful for indirect wording. Low-confidence classifier
        # extras are suppressed when the query already states explicit capabilities.
        explicit = detect_intents(query)

        # A budget request is a composite dependency workflow rather than a single
        # classifier label. Reuse the deterministic planner for this case so the
        # dependent location/hotel/transport/calculator chain remains explicit and
        # testable. The ML router is still used for ordinary capability routing.
        if explicit.get("budget"):
            return super().create_plan(query)

        explicit_labels = {name for name in ["weather","hotel","attractions","transport","restaurant","location"] if explicit.get(name)}
        if explicit_labels:
            # Explicit lexical cues are high-precision. Preserve only extremely
            # confident classifier extras to avoid unrelated calls caused by corpus
            # correlations (for example, travel + currency spuriously implying hotel).
            predicted = explicit_labels | {label for label in predicted if scores.get(label, 0.0) >= 0.985}
        if explicit.get("exclude_hotel"):
            predicted.discard("hotel")
        fx = currency_request_enhanced(query)
        steps: list[PlanStep] = []

        def add(tool: str, reason: str, args: dict[str, Any], deps=None):
            steps.append(PlanStep(len(steps)+1, tool, reason, args, deps or []))

        if city and "location" in predicted:
            add("get_location_info", "ML router predicted destination metadata intent.", {"city": city})
        if city and "weather" in predicted:
            add("get_weather", "ML router predicted weather intent.", {"city": city, "days": days, "unit": "celsius"})
        if fx:
            amount, src, dst = fx
            add("convert_currency", "Explicit currency-conversion pattern detected.", {"amount": amount, "from_currency": src, "to_currency": dst})
        if city and "hotel" in predicted:
            price, rating, top_k = hotel_constraints(query)
            add("search_hotels", "ML router predicted accommodation intent.", {"city": city, "nights": days, "max_price_per_night_eur": price, "min_rating": rating, "top_k": top_k})
        if city and "attractions" in predicted:
            cats, max_ticket, top_k = attraction_constraints(query)
            add("search_attractions", "ML router predicted attraction intent.", {"city": city, "categories": cats, "max_ticket_eur": max_ticket, "top_k": top_k})
        if city and "restaurant" in predicted:
            cuisines, max_meal, min_rating, vegetarian, top_k = restaurant_constraints(query)
            add("search_restaurants", "ML router predicted restaurant intent.", {"city": city, "cuisines": cuisines, "max_meal_eur": max_meal, "min_rating": min_rating, "vegetarian_only": vegetarian, "top_k": top_k})
        if city and "transport" in predicted:
            add("get_transport_options", "ML router predicted transport intent.", {"city": city, "days": days})
        return steps

    def run(self, query: str):
        predicted, scores = self.predict_intents(query)
        run = super().run(query)
        run.model = "ml-intent-router"
        run.metadata["methodology"] = "ml_router"
        run.metadata["predicted_intents"] = sorted(predicted)
        run.metadata["intent_probabilities"] = {k: round(v, 4) for k, v in sorted(scores.items())}
        return run

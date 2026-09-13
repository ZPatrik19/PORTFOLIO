from travel_agent.agent.ml_router_agent import MLRouterTravelAgent


def test_ml_router_predicts_multiple_capabilities():
    agent = MLRouterTravelAgent()
    intents, scores = agent.predict_intents(
        "I am visiting Vienna. Will I need an umbrella and where should we have dinner?"
    )
    assert "weather" in intents
    assert "restaurant" in intents
    assert set(scores) >= {"weather", "restaurant"}


def test_ml_router_builds_executable_plan():
    agent = MLRouterTravelAgent()
    run = agent.run("Find somewhere to stay in Vienna for 2 nights and tell me how to get around without a car.")
    assert "search_hotels" in run.tool_names
    assert "get_transport_options" in run.tool_names
    assert all(c.success for c in run.trace)

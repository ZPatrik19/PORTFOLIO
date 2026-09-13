from travel_agent.agent import PlanThenExecuteTravelAgent

def test_complex_research_query_selects_multiple_tools():
    run=PlanThenExecuteTravelAgent().run("I am going to Vienna for 3 days. Check weather, find hotels under 150 EUR, show museums and landmarks, and public transport costs.")
    assert run.tool_names==["get_weather","search_hotels","search_attractions","get_transport_options"]
    assert run.trace[1].arguments["max_price_per_night_eur"]==150.0
    assert run.trace[2].arguments["categories"]==["museum","landmark"]

def test_written_number_days():
    run=PlanThenExecuteTravelAgent().run("I will spend two days in Vienna. Check the weather.")
    assert run.trace[0].arguments["days"]==2

def test_currency_symbol_before_amount():
    run=PlanThenExecuteTravelAgent().run("How much is €300 in Hungarian forints?")
    assert run.tool_names==["convert_currency"]
    assert run.trace[0].arguments=={"amount":300.0,"from_currency":"EUR","to_currency":"HUF"}


def test_hungarian_multitool_query_with_declined_city_name():
    run = PlanThenExecuteTravelAgent(language="hu").run(
        "3 napra megyek Bécsbe. Nézd meg az időjárást, keress 150 euró alatti hotelt, "
        "ajánlj éttermet és látnivalókat, valamint mondd meg a tömegközlekedési lehetőségeket."
    )
    assert run.tool_names == [
        "get_weather",
        "search_hotels",
        "search_attractions",
        "search_restaurants",
        "get_transport_options",
    ]
    assert all(call.success for call in run.trace)
    assert run.trace[0].arguments["city"] == "Vienna"
    assert run.trace[1].arguments["max_price_per_night_eur"] == 150.0

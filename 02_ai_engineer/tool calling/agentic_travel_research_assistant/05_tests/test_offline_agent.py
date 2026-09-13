from travel_agent.agent import OfflineTravelAgent

def test_no_tool_query_avoids_calls():
    run=OfflineTravelAgent().run("Write a short travel motto.")
    assert run.trace==[]

def test_destination_catalog_supports_many_cities():
    run=OfflineTravelAgent().run("Give me destination information for Barcelona.")
    assert run.tool_names==["get_location_info"]
    assert "Spain" in run.answer

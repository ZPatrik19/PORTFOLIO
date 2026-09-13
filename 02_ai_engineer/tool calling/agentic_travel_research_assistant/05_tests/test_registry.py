from travel_agent.tools import ToolRegistry

def test_registry_contains_expected_tools():
    assert set(ToolRegistry().names)=={"get_location_info","get_weather","convert_currency","search_hotels","search_attractions","search_restaurants","get_transport_options","calculate"}

def test_registry_validates_arguments():
    output,_,success,error=ToolRegistry().execute("get_weather",{"city":"Vienna","days":99,"unit":"celsius"})
    assert not success and "error" in output and error

def test_all_schemas_are_function_tools():
    schemas=ToolRegistry().schemas();assert len(schemas)==8
    assert all(s["type"]=="function" and s["strict"] is True for s in schemas)

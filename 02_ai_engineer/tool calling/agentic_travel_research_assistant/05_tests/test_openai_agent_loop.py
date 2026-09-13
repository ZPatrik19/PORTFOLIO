import json
from types import SimpleNamespace
from travel_agent.agent.openai_agent import OpenAITravelAgent
from travel_agent.tools.currency import CurrencyTool

class FakeResponses:
    def __init__(self):self.calls=[]
    def create(self,**kwargs):
        self.calls.append(kwargs)
        if len(self.calls)==1:
            call=SimpleNamespace(type="function_call",name="convert_currency",arguments=json.dumps({"amount":500,"from_currency":"EUR","to_currency":"HUF"}),call_id="call_1")
            return SimpleNamespace(id="resp_1",output=[call],output_text="")
        return SimpleNamespace(id="resp_2",output=[],output_text="Conversion completed from the tool output.")
class FakeClient:
    def __init__(self):self.responses=FakeResponses()

def test_openai_agent_executes_function_call(monkeypatch):
    def fail(*args,**kwargs):raise OSError("offline")
    monkeypatch.setattr(CurrencyTool,"_live",staticmethod(fail))
    client=FakeClient();agent=OpenAITravelAgent(api_key=None,model="fake",client=client);run=agent.run("Convert 500 EUR to HUF")
    assert run.tool_names==["convert_currency"] and run.trace[0].output["converted_amount"]==197500.0
    returned=client.responses.calls[1]["input"][0];assert returned["type"]=="function_call_output" and returned["call_id"]=="call_1"

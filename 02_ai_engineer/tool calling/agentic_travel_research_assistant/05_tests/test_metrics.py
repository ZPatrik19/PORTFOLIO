from travel_agent.evaluation.metrics import score_case


def test_perfect_case_scores_one():
    expected = [{"name": "get_weather", "arguments": {"city": "Vienna", "days": 3}}]
    actual = [{"name": "get_weather", "arguments": {"city": "Vienna", "days": 3}, "success": True}]
    s = score_case(expected, actual, "ok")
    assert s["tool_selection_accuracy"] == 1
    assert s["argument_accuracy"] == 1
    assert s["task_success"] == 1


def test_unnecessary_call_is_penalized():
    expected = [{"name": "get_weather", "arguments": {"city": "Vienna"}}]
    actual = [
        {"name": "get_weather", "arguments": {"city": "Vienna"}, "success": True},
        {"name": "calculate", "arguments": {"expression": "1+1"}, "success": True},
    ]
    s = score_case(expected, actual, "ok")
    assert s["tool_selection_accuracy"] == 0
    assert s["unnecessary_tool_call_rate"] == 0.5

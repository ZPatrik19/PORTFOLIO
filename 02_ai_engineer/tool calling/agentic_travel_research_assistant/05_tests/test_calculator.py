import pytest
from travel_agent.tools.calculator import CalculatorInput, CalculatorTool


def test_calculator_basic_arithmetic():
    out = CalculatorTool().execute(CalculatorInput(expression="100 + 3 * (9 + 42)"))
    assert out["result"] == 253.0


def test_calculator_rejects_function_calls():
    with pytest.raises(ValueError):
        CalculatorTool().execute(CalculatorInput(expression="__import__('os').system('echo hacked')"))

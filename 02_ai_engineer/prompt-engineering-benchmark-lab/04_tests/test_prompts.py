"""EN: Core prompt-strategy rendering and structured-output strategy behavior.

HU: Az alap promptstratégiák renderelését és a structured-output stratégia működését ellenőrzi.
"""

from prompt_benchmark.constants import LABELS
from prompt_benchmark.prompts import get_strategy, list_strategies


def test_all_strategies_render_ticket():
    """EN: Ensures every built-in strategy can render the ticket without losing user input.

    HU: Biztosítja, hogy minden beépített promptstratégia rendereli a ticketet és nem veszti el a user inputot.
    """
    for name in list_strategies():
        payload = get_strategy(name).build("Please cancel my subscription")
        assert "Please cancel my subscription" in payload.input_text
        assert payload.strategy_name == name


def test_structured_output_strategy():
    """EN: Checks that the structured-output strategy advertises and produces the expected output contract.

    HU: Ellenőrzi, hogy a structured-output stratégia a várt output contractot hirdeti és használja.
    """
    payload = get_strategy("p7_structured_output").build("invoice issue")
    assert payload.structured_output is True
    assert payload.output_mode == "json"

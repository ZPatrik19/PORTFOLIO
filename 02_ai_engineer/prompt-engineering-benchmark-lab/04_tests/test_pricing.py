"""EN: Token-based request cost calculation.

HU: A tokenalapú request-költség számítását ellenőrzi.
"""

from prompt_benchmark.utils.pricing import calculate_estimated_cost


def test_cost_calculation():
    """EN: Checks input/output token pricing arithmetic against a known expected cost.

    HU: Ellenőrzi az input/output tokenárazás aritmetikáját ismert várható költséggel.
    """
    assert calculate_estimated_cost(1_000_000, 1_000_000, 2.0, 8.0) == 10.0

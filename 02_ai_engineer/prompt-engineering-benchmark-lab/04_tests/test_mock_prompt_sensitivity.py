"""EN: Realistic case-type coverage and deterministic quality/cost trade-offs in the mock simulator.

HU: A realisztikus esettípus-lefedettséget és a mock szimulátor determinisztikus quality/cost trade-offjait ellenőrzi.
"""

from pathlib import Path

from prompt_benchmark.benchmark.runner import run_strategy
from prompt_benchmark.data.mock_generator import CASE_TYPES, generate_mock_support_tickets
from prompt_benchmark.data.prepare_benchmark import prepare_splits
from prompt_benchmark.evaluation.metrics import classification_metrics
from prompt_benchmark.llm.client import MockLLMClient
from prompt_benchmark.prompts import get_strategy


def test_mock_dataset_contains_all_realistic_case_types():
    """EN: Checks that the prompt-sensitivity simulator sees the complete set of designed realistic case types.

    HU: Ellenőrzi, hogy a promptérzékenységi szimulátor minden tervezett realisztikus esettípust lefed.
    """
    frame = generate_mock_support_tickets(samples_per_class=120)
    assert set(frame["case_type"].unique()) == set(CASE_TYPES)
    assert set(frame["difficulty"].unique()) == {"easy", "medium", "hard"}
    assert frame["scenario_id"].nunique() == len(frame)


def test_advanced_prompt_is_better_but_more_expensive_in_mock_simulation(tmp_path: Path):
    """EN: Confirms the mock simulator encodes a deterministic quality-versus-token/latency trade-off for advanced prompts.

    HU: Biztosítja, hogy a mock szimulátor determinisztikus quality-versus-token/latency trade-offot modellezzen az advanced promptoknál.
    """
    source = generate_mock_support_tickets(samples_per_class=120)
    benchmark, _, _ = prepare_splits(source, 50, 50, 2, random_seed=42)
    client = MockLLMClient()

    baseline = run_strategy(
        benchmark,
        get_strategy("p0_zero_shot"),
        client,
        tmp_path / "p0.csv",
        0.0,
        0.0,
        force=True,
    )
    advanced = run_strategy(
        benchmark,
        get_strategy("p16_full_advanced_template"),
        client,
        tmp_path / "p16.csv",
        0.0,
        0.0,
        force=True,
    )
    baseline_metrics = classification_metrics(baseline)
    advanced_metrics = classification_metrics(advanced)

    assert advanced_metrics["macro_f1"] > baseline_metrics["macro_f1"]
    assert advanced_metrics["mean_total_tokens"] > baseline_metrics["mean_total_tokens"]
    assert advanced["token_source"].eq("estimated_mock").all()
    assert advanced["latency_source"].eq("simulated_mock").all()

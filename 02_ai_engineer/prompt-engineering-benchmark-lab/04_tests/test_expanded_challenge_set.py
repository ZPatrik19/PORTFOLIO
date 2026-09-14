"""EN: Scale and scenario diversity of the expanded adversarial Challenge Set.

HU: A kibővített, adversarial Challenge Set méretét és scenario-diverzitását ellenőrzi.
"""

from prompt_benchmark.data.mock_generator import CASE_TYPES, generate_mock_support_tickets


def test_expanded_challenge_has_eighteen_scenario_families():
    """EN: Ensures the expanded challenge benchmark contains all 18 designed robustness scenario families.

    HU: Ellenőrzi, hogy a kibővített challenge benchmark mind a 18 tervezett robustness scenario családot tartalmazza.
    """
    frame = generate_mock_support_tickets(samples_per_class=36)
    assert len(CASE_TYPES) == 18
    assert frame["case_type"].nunique() == 18
    assert {
        "negation_correction",
        "quoted_thread",
        "multilingual_mixed",
        "telegraphic_short",
        "primary_last",
        "conditional_distractor",
        "code_log_noise",
        "label_word_attack",
        "double_negation",
    }.issubset(set(frame["case_type"]))


def test_default_generator_is_10x_scale():
    """EN: Ensures the default synthetic generator produces the intended 10x-scale dataset.

    HU: Biztosítja, hogy az alapértelmezett szintetikus generátor a tervezett 10× méretű adathalmazt hozza létre.
    """
    frame = generate_mock_support_tickets()
    assert len(frame) == 10800
    assert frame.groupby("label").size().min() == 1800

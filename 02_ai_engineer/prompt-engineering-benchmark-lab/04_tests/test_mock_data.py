"""EN: Balance and minimum scale of the offline synthetic dataset generator.

HU: Az offline szintetikus adatszett-generátor kiegyensúlyozottságát és minimális méretét ellenőrzi.
"""

from prompt_benchmark.constants import LABELS
from prompt_benchmark.data.mock_generator import generate_mock_support_tickets


def test_mock_generator_is_balanced_and_large_enough_for_default_splits():
    """EN: Ensures the synthetic source is balanced and large enough to create the configured development/holdout/few-shot sets.

    HU: Ellenőrzi, hogy a szintetikus forrás kiegyensúlyozott és elég nagy a konfigurált split-ekhez.
    """
    frame = generate_mock_support_tickets(samples_per_class=120)
    counts = frame["label"].value_counts().to_dict()
    assert len(frame) == 720
    assert set(counts) == set(LABELS)
    assert all(counts[label] == 120 for label in LABELS)
    assert frame["text"].nunique() == len(frame)

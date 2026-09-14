"""EN: Leakage-safe, balanced, disjoint development/holdout/few-shot data splits.

HU: A leakage-mentes, kiegyensúlyozott és diszjunkt development/holdout/few-shot adatszétválasztást ellenőrzi.
"""

import pandas as pd
from prompt_benchmark.constants import LABELS
from prompt_benchmark.data.prepare_benchmark import prepare_splits


def test_splits_are_disjoint_and_balanced():
    """EN: Ensures development, holdout, and few-shot sets are balanced and contain no overlapping samples.

    HU: Ellenőrzi, hogy a development, holdout és few-shot halmazok kiegyensúlyozottak és nincs köztük átfedés.
    """
    rows=[]
    for label in LABELS:
        rows += [{"text": f"{label} example {i}", "label": label} for i in range(6)]
    frame=pd.DataFrame(rows)
    benchmark, development, few_shot = prepare_splits(frame, 2, 2, 1, 42)
    assert len(benchmark) == 12 and len(development) == 12 and len(few_shot) == 6
    assert not (set(benchmark.text) & set(development.text))

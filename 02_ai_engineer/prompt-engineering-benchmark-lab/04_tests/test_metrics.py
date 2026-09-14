"""EN: Classification metrics, invalid predictions, and uncertainty for small perfect samples.

HU: A klasszifikációs metrikákat, invalid predikciók kezelését és a kis tökéletes minták bizonytalanságát ellenőrzi.
"""

import pandas as pd
from prompt_benchmark.evaluation.metrics import classification_metrics, wilson_interval


def test_perfect_metrics():
    """EN: Checks the metric implementation against a perfect six-class prediction case.

    HU: Ellenőrzi a metric implementációt egy tökéletes, hatosztályos predikciós példán.
    """
    frame = pd.DataFrame({
        "true_label": ["api", "billing", "cancellation", "complaint", "technical", "upgrade"],
        "predicted_label": ["api", "billing", "cancellation", "complaint", "technical", "upgrade"],
        "valid_output": [True]*6, "valid_json": [None]*6,
        "input_tokens": [10]*6, "output_tokens": [1]*6, "total_tokens": [11]*6,
        "latency_seconds": [0.1]*6, "estimated_cost_usd": [0.0]*6, "error": [None]*6,
    })
    metrics = classification_metrics(frame)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_confusion_table_handles_invalid_nan_predictions():
    """EN: Ensures invalid/NaN predictions are represented safely in the confusion table instead of crashing sklearn.

    HU: Biztosítja, hogy az invalid/NaN predikciók külön kezelhetők legyenek és ne omoljon össze a confusion matrix.
    """
    import numpy as np
    import pandas as pd
    from prompt_benchmark.constants import INVALID_LABEL, LABELS
    from prompt_benchmark.evaluation.metrics import confusion_table

    frame = pd.DataFrame({
        "true_label": [LABELS[0], LABELS[1], LABELS[1]],
        "predicted_label": [LABELS[0], np.nan, LABELS[0]],
    })
    table = confusion_table(frame)
    assert list(table.columns) == [*LABELS, INVALID_LABEL]
    assert table.loc[LABELS[1], INVALID_LABEL] == 1
    assert int(table.to_numpy().sum()) == 3


def test_wilson_interval_keeps_small_perfect_sample_uncertain():
    """EN: Ensures a tiny 100% pilot still shows statistical uncertainty instead of implying proven perfection.

    HU: Biztosítja, hogy egy kis 100%-os pilot továbbra is bizonytalanságot mutasson, ne bizonyított tökéletességet sugalljon.
    """
    low, high = wilson_interval(10, 10)
    assert low < 0.8
    assert high > 0.999

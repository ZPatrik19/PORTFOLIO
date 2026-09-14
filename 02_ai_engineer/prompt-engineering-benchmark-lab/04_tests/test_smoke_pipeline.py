"""EN: Minimal end-to-end offline benchmark execution through validation, prompting, inference, and metrics.

HU: Minimális end-to-end offline benchmarkot futtat validációtól a promptoláson és inference-en át a metrikákig.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from prompt_benchmark.benchmark.runner import run_strategy
from prompt_benchmark.evaluation.metrics import classification_metrics
from prompt_benchmark.llm.client import MockLLMClient
from prompt_benchmark.prompts import get_strategy


def test_minimal_end_to_end_benchmark_smoke(tmp_path: Path) -> None:
    """EN: Runs the smallest complete offline path from validated data through prompt rendering, mock inference, checkpointing, and evaluation.

    HU: A legkisebb teljes offline útvonalat futtatja validált adattól prompt renderen és mock inference-en át checkpointig és evaluációig.
    """
    benchmark = pd.DataFrame(
        [
            {"sample_id": "s0", "text": "API returns 429.", "true_label": "api"},
            {"sample_id": "s1", "text": "I was charged twice.", "true_label": "billing"},
            {"sample_id": "s2", "text": "Cancel before renewal.", "true_label": "cancellation"},
            {"sample_id": "s3", "text": "I am unhappy with the service.", "true_label": "complaint"},
            {"sample_id": "s4", "text": "The app crashes on startup.", "true_label": "technical"},
            {"sample_id": "s5", "text": "Upgrade us to the larger plan.", "true_label": "upgrade"},
        ]
    )
    result = run_strategy(
        benchmark,
        get_strategy("p0_zero_shot"),
        MockLLMClient(),
        tmp_path / "smoke.csv",
        0.0,
        0.0,
        force=True,
    )
    metrics = classification_metrics(result)
    assert len(result) == 6
    assert metrics["api_error_count"] == 0
    assert 0.0 <= metrics["macro_f1"] <= 1.0

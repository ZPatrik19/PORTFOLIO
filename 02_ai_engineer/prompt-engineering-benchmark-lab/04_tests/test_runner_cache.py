"""EN: Benchmark-cache correctness across changed data, profiles, sampling settings, and pricing updates.

HU: A benchmark cache helyességét ellenőrzi megváltozott adatok, profilok, sampling beállítások és pricing esetén.
"""

from pathlib import Path

import pandas as pd
import pytest

from prompt_benchmark.benchmark.runner import run_strategy
from prompt_benchmark.llm.client import MockLLMClient
from prompt_benchmark.prompts import get_strategy


def test_cache_rejects_same_sample_id_with_different_text(tmp_path: Path):
    """EN: Prevents stale cached predictions from being reused when a sample ID now maps to different content.

    HU: Megakadályozza stale cache predikció használatát, ha ugyanaz a sample_id már más tartalomra mutat.
    """
    path = tmp_path / "p0.csv"
    first = pd.DataFrame([
        {"sample_id": "bench_0000", "text": "Please cancel my plan.", "true_label": "cancellation"}
    ])
    second = pd.DataFrame([
        {"sample_id": "bench_0000", "text": "I was charged twice.", "true_label": "billing"}
    ])
    strategy = get_strategy("p0_zero_shot")
    client = MockLLMClient()
    run_strategy(first, strategy, client, path, 0.0, 0.0)
    with pytest.raises(ValueError, match="different data"):
        run_strategy(second, strategy, client, path, 0.0, 0.0)


def _six_row_benchmark() -> pd.DataFrame:
    labels = ["api", "billing", "cancellation", "complaint", "technical", "upgrade"]
    return pd.DataFrame(
        [
            {"sample_id": f"bench_{i}", "text": f"{label} example {i}", "true_label": label}
            for i, label in enumerate(labels)
        ]
    )


def test_cache_rejects_rows_outside_changed_run_profile(tmp_path: Path):
    """EN: Prevents a full-run cache from contaminating a smaller/different pilot profile.

    HU: Megakadályozza, hogy egy full-run cache beszennyezzen egy eltérő pilot profilt.
    """
    path = tmp_path / "p0.csv"
    benchmark = _six_row_benchmark()
    strategy = get_strategy("p0_zero_shot")
    client = MockLLMClient()
    run_strategy(benchmark, strategy, client, path, 0.0, 0.0, force=True)
    with pytest.raises(ValueError, match="outside the current benchmark selection"):
        run_strategy(benchmark, strategy, client, path, 0.0, 0.0, limit=3)


def test_cache_rejects_sampling_setting_change_to_default(tmp_path: Path):
    """EN: Invalidates cached predictions when decoding settings change, including transitions back to defaults.

    HU: Sampling beállítás változásakor invalidálja a cache-t, beleértve az alapértékre való visszaállást is.
    """
    path = tmp_path / "p0.csv"
    benchmark = _six_row_benchmark()
    strategy = get_strategy("p0_zero_shot")
    first_client = MockLLMClient(temperature=0.7)
    run_strategy(benchmark, strategy, first_client, path, 0.0, 0.0, force=True)
    default_client = MockLLMClient(temperature=None)
    with pytest.raises(ValueError, match="Cached temperature"):
        run_strategy(benchmark, strategy, default_client, path, 0.0, 0.0)


def test_cached_cost_is_recomputed_when_pricing_changes(tmp_path: Path):
    """EN: Reuses valid predictions but recomputes monetary cost when pricing configuration changes.

    HU: A valid predikció cache-t újrahasználja, de pricing változáskor újraszámolja a költséget.
    """
    path = tmp_path / "p0.csv"
    benchmark = _six_row_benchmark()
    strategy = get_strategy("p0_zero_shot")
    client = MockLLMClient()
    first = run_strategy(benchmark, strategy, client, path, 1.0, 1.0, force=True)
    second = run_strategy(benchmark, strategy, client, path, 2.0, 2.0)
    assert second["estimated_cost_usd"].sum() == pytest.approx(first["estimated_cost_usd"].sum() * 2.0)


def test_cache_accepts_numeric_equivalent_sampling_settings(tmp_path):
    """EN: Treats numerically equivalent settings such as 40 and 40.0 as the same cache identity.

    HU: A numerikusan ekvivalens 40 és 40.0 értékeket azonos cache-identitásként kezeli.
    """
    """CSV dtype coercion (40 -> 40.0) must not invalidate an otherwise identical cache."""
    import pandas as pd

    from prompt_benchmark.benchmark.runner import run_strategy
    from prompt_benchmark.llm.client import MockLLMClient
    from prompt_benchmark.prompts import get_strategy

    benchmark = pd.DataFrame({
        "sample_id": ["s1"],
        "text": ["Please cancel my subscription."],
        "true_label": ["cancellation"],
    })
    output = tmp_path / "cache.csv"
    client = MockLLMClient(top_k=40)
    run_strategy(benchmark, get_strategy("p0_zero_shot"), client, output, 0.0, 0.0, force=True)

    cached = pd.read_csv(output)
    cached["top_k"] = cached["top_k"].astype(float)
    cached.to_csv(output, index=False)

    resumed = run_strategy(benchmark, get_strategy("p0_zero_shot"), client, output, 0.0, 0.0)
    assert len(resumed) == 1

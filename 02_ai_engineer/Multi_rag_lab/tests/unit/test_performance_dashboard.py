import sys
import types

# Keep this unit test dependency-light. In the real UI environment Streamlit is
# installed; CI can still validate the dataframe contract without importing it.
sys.modules.setdefault("streamlit", types.ModuleType("streamlit"))

from ui.components.performance import _frame


def test_performance_frame_has_unique_localized_metric_columns() -> None:
    frame = _frame(
        [
            {
                "component": "embedding",
                "device": "cuda",
                "total_ms": 20.0,
                "mean_ms": 4.0,
                "median_ms": 3.8,
                "p95_ms": 5.5,
                "throughput_per_sec": 250.0,
                "workload_size": 5,
            }
        ]
    )

    assert frame.columns.is_unique
    chart_frame = frame[["Komponens", "Eszköz", "Átlag ms", "P95 ms", "Áteresztőképesség/s"]].copy()
    assert chart_frame.columns.is_unique
    assert chart_frame.columns.tolist() == [
        "Komponens",
        "Eszköz",
        "Átlag ms",
        "P95 ms",
        "Áteresztőképesség/s",
    ]

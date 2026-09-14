"""EN: Regression tests for A/B Playground token reshaping and the previous pandas.melt column collision.

HU: Regressziós teszteket tartalmaz az A/B Playground token-átalakítására és a korábbi pandas.melt oszlopütközésre.
"""

from __future__ import annotations

import pandas as pd

from prompt_benchmark.ui.playground_data import reshape_token_usage


def test_reshape_token_usage_allows_existing_total_tokens_column() -> None:
    """EN: Regression test for the former pandas.melt value_name="tokens" collision in classification A/B charts.

    HU: Regressziós teszt a korábbi pandas.melt value_name="tokens" oszlopütközésre a klasszifikációs A/B chartban.
    """
    source = pd.DataFrame(
        [
            {
                "variant": "A",
                "strategy": "P1",
                "tokens": 180,
                "input_tokens": 176,
                "output_tokens": 4,
            },
            {
                "variant": "B",
                "strategy": "P16",
                "tokens": 2098,
                "input_tokens": 2090,
                "output_tokens": 8,
            },
        ]
    )

    long_df = reshape_token_usage(source, id_vars=["variant", "strategy"])

    assert "token_count" in long_df.columns
    assert "tokens" not in long_df.columns
    assert len(long_df) == 4
    assert long_df["token_count"].sum() == 2278
    assert set(long_df["token_type"]) == {"input_tokens", "output_tokens"}


def test_reshape_token_usage_generation_shape() -> None:
    """EN: Checks the long-form token dataframe used by generative A/B charts.

    HU: Ellenőrzi a generatív A/B chartokhoz használt long-form token DataFrame alakját.
    """
    source = pd.DataFrame(
        [
            {"variant": "A", "template": "T0", "tokens": 12, "input_tokens": 8, "output_tokens": 4},
            {"variant": "B", "template": "T5", "tokens": 30, "input_tokens": 20, "output_tokens": 10},
        ]
    )

    long_df = reshape_token_usage(source, id_vars=["variant", "template"])

    assert list(long_df.columns) == ["variant", "template", "token_type", "token_count"]
    assert long_df.groupby("variant")["token_count"].sum().to_dict() == {"A": 12, "B": 30}

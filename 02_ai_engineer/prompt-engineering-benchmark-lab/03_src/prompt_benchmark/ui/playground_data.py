from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def reshape_token_usage(
    result_df: pd.DataFrame,
    *,
    id_vars: Sequence[str],
) -> pd.DataFrame:
    """Return a long-form input/output token table for Plotly charts.

    The source benchmark rows already contain a ``tokens`` column for total
    token usage. Pandas ``melt`` rejects creating another column with the same
    name, so the long-form value column is deliberately named
    ``token_count``.
    """
    required = [*id_vars, "input_tokens", "output_tokens"]
    missing = [column for column in required if column not in result_df.columns]
    if missing:
        raise ValueError(f"Missing token chart columns: {missing}")

    return result_df.loc[:, required].melt(
        id_vars=list(id_vars),
        value_vars=["input_tokens", "output_tokens"],
        var_name="token_type",
        value_name="token_count",
    )

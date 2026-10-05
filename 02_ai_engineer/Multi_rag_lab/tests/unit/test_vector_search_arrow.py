from __future__ import annotations

import pandas as pd


def test_nullable_rank_columns_do_not_mix_strings_and_integers() -> None:
    frame = pd.DataFrame({
        "Dense rank": [1, None, 3],
        "BM25 rank": [None, 2, 4],
    })
    frame["Dense rank"] = pd.array(frame["Dense rank"], dtype="Int64")
    frame["BM25 rank"] = pd.array(frame["BM25 rank"], dtype="Int64")
    assert str(frame["Dense rank"].dtype) == "Int64"
    assert str(frame["BM25 rank"].dtype) == "Int64"
    assert pd.isna(frame.loc[1, "Dense rank"])
    assert pd.isna(frame.loc[0, "BM25 rank"])

    try:
        import pyarrow as pa
    except ImportError:
        return
    table = pa.Table.from_pandas(frame, preserve_index=False)
    assert table.num_rows == 3

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ui.components.tables import arrow_safe_frame


ROOT = Path(__file__).resolve().parents[2]


def test_arrow_safe_frame_normalizes_mixed_display_columns() -> None:
    frame = pd.DataFrame(
        {
            "Érték": [0, "nincs", "CPU", None],
            "Rank": [1, 2, None, 4],
            "Bool": [True, False, None, True],
        }
    )
    safe = arrow_safe_frame(frame)

    assert str(safe["Érték"].dtype) == "string"
    assert safe.loc[0, "Érték"] == "0"
    assert safe.loc[1, "Érték"] == "nincs"
    assert str(safe["Rank"].dtype) in {"float64", "Float64", "Int64"}

    try:
        import pyarrow as pa
    except ImportError:
        return
    table = pa.Table.from_pandas(safe, preserve_index=False)
    assert table.num_rows == 4


def test_all_streamlit_tables_use_arrow_safe_wrappers() -> None:
    offenders: list[str] = []
    for path in (ROOT / "ui").rglob("*.py"):
        if path.name == "tables.py":
            continue
        source = path.read_text(encoding="utf-8")
        if "st.dataframe(" in source or "st.data_editor(" in source:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []

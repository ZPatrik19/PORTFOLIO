from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import streamlit as st


def render_dataframe_exports(frame: pd.DataFrame, *, stem: str, key_prefix: str) -> None:
    if frame.empty:
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    c1, c2 = st.columns(2)
    c1.download_button(
        "CSV letöltése",
        frame.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{stem}_{stamp}.csv",
        mime="text/csv",
        key=f"{key_prefix}_csv",
        width="stretch",
    )
    c2.download_button(
        "JSON letöltése",
        json.dumps(frame.to_dict(orient="records"), ensure_ascii=False, indent=2, default=str).encode("utf-8"),
        file_name=f"{stem}_{stamp}.json",
        mime="application/json",
        key=f"{key_prefix}_json",
        width="stretch",
    )


def render_text_export(text: str, *, label: str, filename: str, key: str) -> None:
    st.download_button(label, text.encode("utf-8"), file_name=filename, mime="text/plain", key=key)

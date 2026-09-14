from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import plotly.express as px
import streamlit as st

from prompt_benchmark.paths import PATHS

from prompt_benchmark.ui.run_history import list_runs, load_manifest, load_summary, run_dir


def render_workflow_banner(language: str, provider_ready: bool) -> None:
    hu = language == "hu"
    dataset_ready = (PATHS.processed_data / "benchmark.csv").exists()
    has_prompt = bool(list(PATHS.custom_prompts.glob("*.json"))) or bool(list(PATHS.playground_prompts.glob("*.json")))
    runs = list_runs(status="complete")
    benchmark_ready = not runs.empty
    steps = [
        ("1. Provider/API", provider_ready, "Bal sidebar" if hu else "Left sidebar"),
        ("2. Dataset", dataset_ready, "Dataset"),
        ("3. Playground", has_prompt, "Playground"),
        ("4. Benchmark", benchmark_ready, "Benchmark"),
        ("5. Output validation", benchmark_ready, "Output Validation"),
        ("6. Dashboard", benchmark_ready, "Dashboard"),
        ("7. History", benchmark_ready, "History"),
    ]
    st.markdown("### 🧭 " + ("Ajánlott munkafolyamat" if hu else "Recommended workflow"))
    cols = st.columns(len(steps))
    for col, (name, done, where) in zip(cols, steps):
        col.markdown(f"**{'✅' if done else '○'} {name}**")
        col.caption(where)
    st.caption(
        "A lépések állapota és a benchmark history lemezre mentődik, ezért oldalváltás és UI újraindítás után is visszatölthető."
        if hu else
        "Step state and benchmark history are persisted to disk, so runs remain available after navigation and UI restarts."
    )


def render_history_panel(language: str, provider: str, strategy_title: Callable[[str], str]) -> None:
    hu = language == "hu"
    st.subheader("Benchmark history és futások összehasonlítása" if hu else "Benchmark history and run comparison")
    runs = list_runs(provider=provider, status=None)
    if runs.empty:
        st.info("Még nincs mentett benchmark futás." if hu else "No benchmark runs have been saved yet.")
        return
    show_cols = [c for c in ["run_id", "status", "completed_at", "provider", "model", "suite", "sample_count_per_strategy", "strategies", "best_strategy", "best_macro_f1", "total_tokens", "elapsed_wall_seconds"] if c in runs]
    st.dataframe(runs[show_cols], use_container_width=True, hide_index=True)
    completed = runs[runs["status"] == "complete"].copy() if "status" in runs else runs.copy()
    if completed.empty:
        return
    labels = []
    id_by_label = {}
    for row in completed.itertuples(index=False):
        rid = str(getattr(row, "run_id"))
        label = f"{rid} · {getattr(row, 'suite', '')} · F1={getattr(row, 'best_macro_f1', 0):.3f}"
        labels.append(label)
        id_by_label[label] = rid
    current = st.session_state.get("selected_history_run_id")
    default_idx = 0
    if current:
        for i, label in enumerate(labels):
            if id_by_label[label] == current:
                default_idx = i
                break
    selected_label = st.selectbox("Futás megnyitása" if hu else "Open run", labels, index=default_idx, key=f"{language}_history_run_select_v3")
    run_id = id_by_label[selected_label]
    st.session_state["selected_history_run_id"] = run_id
    summary = load_summary(run_id)
    manifest = load_manifest(run_id)
    if summary is not None and not summary.empty:
        summary = summary.copy()
        summary["strategy_title"] = summary["strategy"].map(strategy_title)
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(summary.sort_values("macro_f1", ascending=False), x="strategy_title", y="macro_f1", range_y=[0, 1], title="Macro F1"), use_container_width=True)
        c2.plotly_chart(px.scatter(summary, x="mean_total_tokens", y="macro_f1", size="p95_latency_seconds", hover_name="strategy_title", title="Quality vs tokens"), use_container_width=True)
        st.dataframe(summary, use_container_width=True, hide_index=True)
    with st.expander("Run manifest / beállítások", expanded=False):
        st.json(manifest)
    root = run_dir(run_id)
    dataset_path = root / "dataset_used.csv"
    if dataset_path.exists():
        st.download_button(
            "Dataset snapshot letöltése" if hu else "Download dataset snapshot",
            data=dataset_path.read_bytes(),
            file_name=f"{run_id}_dataset.csv",
            mime="text/csv",
            key=f"{language}_history_dataset_download_v3",
        )

    st.markdown("### " + ("Futások összehasonlítása" if hu else "Compare runs"))
    compare_labels = st.multiselect(
        "Válassz legfeljebb 5 futást" if hu else "Choose up to 5 runs",
        labels,
        default=labels[: min(3, len(labels))],
        max_selections=5,
        key=f"{language}_history_compare_v3",
    )
    compare_rows = []
    for label in compare_labels:
        rid = id_by_label[label]
        s = load_summary(rid)
        if s is None or s.empty:
            continue
        best = s.sort_values("macro_f1", ascending=False).iloc[0]
        m = load_manifest(rid)
        compare_rows.append({
            "run_id": rid,
            "suite": m.get("suite"),
            "model": m.get("model"),
            "best_strategy": strategy_title(str(best["strategy"])),
            "best_macro_f1": float(best["macro_f1"]),
            "total_tokens": float(s.get("total_tokens", pd.Series([0])).sum()),
            "wall_seconds": float(m.get("elapsed_wall_seconds", 0) or 0),
        })
    if compare_rows:
        cmp = pd.DataFrame(compare_rows)
        a, b = st.columns(2)
        a.plotly_chart(px.bar(cmp, x="run_id", y="best_macro_f1", color="suite", range_y=[0, 1], title="Best Macro F1 by run"), use_container_width=True)
        b.plotly_chart(px.bar(cmp, x="run_id", y="total_tokens", color="suite", title="Total tokens by run"), use_container_width=True)
        st.dataframe(cmp, use_container_width=True, hide_index=True)

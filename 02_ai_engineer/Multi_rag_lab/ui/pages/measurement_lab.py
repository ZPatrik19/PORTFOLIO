from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from rag_engine.evaluation.load import benchmark_retrieval_load
from rag_engine.evaluation.robustness import evaluate_retriever_robustness
from rag_engine.evaluation.statistics import paired_bootstrap_comparison
from rag_engine.presets import HUNGARIAN_QUERY_PRESETS
from ui.components.common import get_lab
from ui.components.education import kpi_cards, note_box, page_intro, section_intro
from ui.components.tables import safe_dataframe


def _retriever(lab, mode: str):
    return {"dense": lab.dense, "bm25": lab.sparse, "hybrid": lab.hybrid}[mode]


def _latency_samples(retriever, queries: list[str], *, top_k: int, candidate_count: int) -> list[float]:
    import time

    values: list[float] = []
    for query in queries:
        started = time.perf_counter()
        try:
            retriever.retrieve(query, top_k=top_k, candidate_count=candidate_count)
        except TypeError:
            retriever.retrieve(query, top_k=top_k)
        values.append((time.perf_counter() - started) * 1000.0)
    return values


def render() -> None:
    page_intro(
        "Measurement Lab",
        "Statisztikai bizonytalanság, terhelés alatti latency és query-robosztusság ugyanazon aktív RAG indexen. A cél nem egyetlen benchmarkszám, hanem stabil és reprodukálható mérési protokoll.",
        eyebrow="WARM-UP · REPEATS · BOOTSTRAP CI · CONCURRENCY · ROBUSTNESS",
    )

    lab = get_lab()
    top_k = int(st.session_state.get("top_k", 5))
    candidate_count = int(st.session_state.get("candidate_count", 20))
    queries = [preset.query for preset in HUNGARIAN_QUERY_PRESETS]

    load_tab, robustness_tab, compare_tab = st.tabs(["Terhelés és variancia", "Query robosztusság", "Páros A/B statisztika"])

    with load_tab:
        section_intro("Terhelési profil", "Warm-up után ismételt lekérdezésekkel P50/P90/P95/P99, szórás, CV, QPS és bootstrap 95% CI készül.")
        c1, c2, c3 = st.columns(3)
        with c1:
            mode = st.selectbox("Retriever", ["dense", "bm25", "hybrid"], index=2, key="measure_load_retriever")
        with c2:
            repeats = st.slider("Ismétlések", 1, 10, 3, key="measure_repeats")
        with c3:
            warmup = st.slider("Warm-up kérések", 0, 10, 2, key="measure_warmup")
        levels = st.multiselect("Konkurencia szintek", [1, 2, 4, 8], default=[1, 2, 4], key="measure_concurrency")
        if st.button("Terhelési mérés futtatása", type="primary", key="measure_load_run"):
            active = _retriever(lab, mode)
            rows = [
                benchmark_retrieval_load(
                    active,
                    queries,
                    concurrency=level,
                    repeats=repeats,
                    warmup_requests=warmup,
                    top_k=top_k,
                    candidate_count=candidate_count,
                ).to_dict()
                for level in levels
            ]
            st.session_state["measurement_load_rows"] = rows

        rows = st.session_state.get("measurement_load_rows", [])
        if rows:
            frame = pd.DataFrame(rows)
            kpi_cards(
                [
                    ("Legjobb QPS", f"{frame['requests_per_second'].max():.2f}", "Falióra-alapú request throughput."),
                    ("Legjobb P95", f"{frame['p95_latency_ms'].min():.2f} ms", "Alacsonyabb kedvezőbb."),
                    ("Legkisebb CV", f"{frame['latency_cv'].min():.3f}", "Kisebb relatív szórás = stabilabb latency."),
                    ("Mérések", str(int(frame['requests'].sum())), "Warm-up nélkül számolt mért kérések."),
                ],
                columns=4,
            )
            safe_dataframe(frame.round(4), width="stretch", hide_index=True)
            chart = px.line(frame, x="concurrency", y=["p50_latency_ms", "p95_latency_ms", "p99_latency_ms"], markers=True, title="Latency percentilisek konkurencia szerint")
            chart.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(chart, width="stretch", key="measurement_load_percentiles")

    with robustness_tab:
        section_intro("Query-robosztusság", "Ugyanazt a kérdést casing, punctuation, whitespace és ékezet-eltérésekkel futtatja; a Top-K ranking stabilitását méri.")
        preset = st.selectbox("Kérdés", HUNGARIAN_QUERY_PRESETS, format_func=lambda item: f"{item.label} · {item.topic}", key="measure_robust_query")
        mode = st.selectbox("Retriever", ["dense", "bm25", "hybrid"], index=2, key="measure_robust_retriever")
        if st.button("Robosztussági mérés", type="primary", key="measure_robust_run"):
            summary = evaluate_retriever_robustness(_retriever(lab, mode), preset.query, top_k=top_k, candidate_count=candidate_count)
            st.session_state["measurement_robustness"] = summary.to_dict()
        summary = st.session_state.get("measurement_robustness")
        if summary:
            kpi_cards(
                [
                    ("Átlag Jaccard@K", f"{summary['mean_jaccard_at_k']:.3f}", "Top-K halmazstabilitás."),
                    ("Legrosszabb Jaccard@K", f"{summary['worst_jaccard_at_k']:.3f}", "Legérzékenyebb perturbáció."),
                    ("Prefix stabilitás", f"{summary['mean_prefix_stability_at_k']:.3f}", "A korai ranghelyeket is figyelembe veszi."),
                    ("Top-1 megtartás", f"{summary['first_result_retention_rate']:.0%}", "Az eredeti első találat bent maradt-e a Top-K-ban."),
                ],
                columns=4,
            )
            safe_dataframe(pd.DataFrame(summary["observations"]).round(4), width="stretch", hide_index=True)

    with compare_tab:
        section_intro("Páros A/B összehasonlítás", "Ugyanazokon a kérdéseken mért latency-ket páros bootstrap eljárással hasonlítja össze. Ez csökkenti a kérdésmixből eredő zajt.")
        c1, c2 = st.columns(2)
        with c1:
            mode_a = st.selectbox("A konfiguráció", ["dense", "bm25", "hybrid"], index=0, key="measure_a")
        with c2:
            mode_b = st.selectbox("B konfiguráció", ["dense", "bm25", "hybrid"], index=2, key="measure_b")
        if st.button("A/B mérés", type="primary", key="measure_compare_run"):
            a_values = _latency_samples(_retriever(lab, mode_a), queries, top_k=top_k, candidate_count=candidate_count)
            b_values = _latency_samples(_retriever(lab, mode_b), queries, top_k=top_k, candidate_count=candidate_count)
            comparison = paired_bootstrap_comparison(a_values, b_values, higher_is_better=False)
            st.session_state["measurement_comparison"] = {"a": mode_a, "b": mode_b, **comparison.to_dict()}
        comparison = st.session_state.get("measurement_comparison")
        if comparison:
            kpi_cards(
                [
                    ("A átlag", f"{comparison['mean_a']:.2f} ms", str(comparison['a'])),
                    ("B átlag", f"{comparison['mean_b']:.2f} ms", str(comparison['b'])),
                    ("Delta B−A", f"{comparison['mean_delta_b_minus_a']:.2f} ms", f"95% CI: {comparison['ci_low']:.2f}…{comparison['ci_high']:.2f}"),
                    ("Bootstrap támogatás", f"{comparison['probability_b_better']:.1%}", "Empirikus bootstrap arány; nem p-érték."),
                ],
                columns=4,
            )
            note_box("Értelmezés", "A konfidenciaintervallum és a páros összehasonlítás fontosabb, mint egyetlen gyorsulási szám. Rövid futásnál a zaj nagy lehet; stabil benchmarkhoz növeld az ismétlésszámot.")

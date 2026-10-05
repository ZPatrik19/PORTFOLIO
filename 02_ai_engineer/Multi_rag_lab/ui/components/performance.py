from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.charts.performance import (
    device_speedup_chart,
    llm_telemetry_chart,
    performance_latency_chart,
    performance_throughput_chart,
    performance_total_time_chart,
    resource_usage_chart,
)
from ui.components.education import empty_state, kpi_cards, note_box
from ui.components.exports import render_dataframe_exports
from ui.components.tables import safe_dataframe


COMPONENT_HU = {
    "embedding": "Beágyazás",
    "retrieval-dense": "Dense visszakeresés",
    "retrieval-bm25": "BM25 visszakeresés",
    "retrieval-hybrid": "Hibrid visszakeresés",
    "retrieval": "Visszakeresés",
    "reranking-lexical": "Lexikális újrarangsorolás",
    "reranking-cross-encoder": "Cross-Encoder újrarangsorolás",
    "reranking": "Újrarangsorolás",
    "llm-generation": "LLM-generálás",
    "end-to-end-rag": "Teljes RAG folyamat",
}


def _frame(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records).copy()
    frame["Komponens"] = frame["component"].map(COMPONENT_HU).fillna(frame["component"])
    frame["Eszköz"] = frame["device"].fillna("—")
    frame["Összes ms"] = frame.get("total_ms")
    frame["Átlag ms"] = frame.get("mean_ms")
    frame["P95 ms"] = frame.get("p95_ms")
    frame["Áteresztőképesség/s"] = frame.get("throughput_per_sec")
    frame["TTFT ms"] = frame.get("ttft_ms")
    frame["Token/s"] = frame.get("tokens_per_second")
    frame["Kimeneti token"] = frame.get("output_tokens")
    frame["Kontextus token"] = frame.get("context_tokens")
    return frame


def render_performance_dashboard(records: list[dict], metadata: dict[str, object]) -> None:
    frame = _frame(records)
    if frame.empty:
        empty_state(
            "Még nincs teljesítménymérés",
            "Futtasd a benchmarkot a komponensek és a teljes RAG folyamat sebességének összehasonlításához.",
        )
        return

    fastest = frame.loc[frame["mean_ms"].fillna(frame["total_ms"]).idxmin()]
    max_throughput = frame.loc[frame["throughput_per_sec"].idxmax()]
    kpi_cards(
        [
            (
                "Mért komponensek",
                str(frame["Komponens"].nunique()),
                "Beágyazás, visszakeresés, újrarangsorolás, LLM és teljes RAG folyamat is mérhető.",
            ),
            (
                "Legkisebb átlagidő",
                f"{float(fastest.get('mean_ms') or fastest['total_ms']):.1f} ms",
                str(fastest["Komponens"]),
            ),
            (
                "Legnagyobb áteresztőképesség",
                f"{float(max_throughput['throughput_per_sec']):.1f}/s",
                str(max_throughput["Komponens"]),
            ),
            ("Mérési sorok", str(len(frame)), "Minden sor ténylegesen futtatott komponensmérés."),
        ],
        columns=4,
    )

    tabs = st.tabs(
        [
            "Áttekintés",
            "Beágyazás",
            "Visszakeresés",
            "Újrarangsorolás",
            "LLM + teljes folyamat",
            "CPU / CUDA",
            "Erőforrások",
        ]
    )
    with tabs[0]:
        summary = frame[
            ["Komponens", "Eszköz", "total_ms", "mean_ms", "median_ms", "p95_ms", "throughput_per_sec", "workload_size"]
        ].copy()
        summary.columns = [
            "Komponens",
            "Eszköz",
            "Összes ms",
            "Átlag ms",
            "Medián ms",
            "P95 ms",
            "Áteresztőképesség/s",
            "Terhelés",
        ]
        c1, c2 = st.columns(2)
        with c1:
            performance_total_time_chart(summary, key="perf_overview_total")
        with c2:
            performance_throughput_chart(summary, key="perf_overview_throughput")
        with st.expander("Nyers teljesítményeredmények", expanded=False):
            safe_dataframe(summary, width="stretch", hide_index=True)
        render_dataframe_exports(frame, stem="teljesitmenybenchmark", key_prefix="performance_export")
        note_box(
            "Értelmezés",
            "A komponensek terhelése eltérő lehet, ezért a teljes idő mellett az átlagos késleltetést, P95-öt és az áteresztőképességet együtt érdemes figyelni.",
        )

    groups = {
        "Beágyazás": frame[frame["component"] == "embedding"],
        "Visszakeresés": frame[frame["component"].astype(str).str.startswith("retrieval")],
        "Újrarangsorolás": frame[frame["component"].astype(str).str.startswith("reranking")],
        "LLM + teljes folyamat": frame[frame["component"].isin(["llm-generation", "end-to-end-rag"])],
    }
    for tab, name in zip(tabs[1:5], groups, strict=False):
        with tab:
            data = groups[name]
            if data.empty:
                empty_state(
                    f"Nincs {name.lower()} mérés", "A jelenlegi benchmark konfiguráció ezt a komponenst nem futtatta."
                )
                continue
            safe_dataframe(
                data[
                    [
                        "Komponens",
                        "Eszköz",
                        "total_ms",
                        "mean_ms",
                        "median_ms",
                        "p95_ms",
                        "throughput_per_sec",
                        "workload_size",
                        "cold_start_ms",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )
            # ``_frame`` already exposes localized metric columns.  Renaming the
            # raw columns on the full frame created duplicate labels (e.g. two
            # ``Átlag ms`` columns), which Plotly/Narwhals rejects.  Select only
            # the canonical localized chart schema instead.
            chart_frame = data[["Komponens", "Eszköz", "Átlag ms", "P95 ms", "Áteresztőképesség/s"]].copy()
            c1, c2 = st.columns(2)
            with c1:
                performance_latency_chart(chart_frame, key=f"perf_latency_{name}")
            with c2:
                performance_throughput_chart(chart_frame, key=f"perf_throughput_{name}")
            if name == "LLM + teljes folyamat":
                telemetry_frame = data[["Komponens", "TTFT ms", "Token/s", "Kimeneti token", "Kontextus token"]].copy()
                llm_telemetry_chart(telemetry_frame, key="perf_llm_telemetry")

    with tabs[5]:
        speedup_frame = frame[["Komponens", "Eszköz", "total_ms", "throughput_per_sec"]].copy()
        speedup_frame.columns = ["Komponens", "Eszköz", "Összes ms", "Áteresztőképesség/s"]
        device_speedup_chart(speedup_frame, key="perf_device_speedup")
        note_box(
            "CPU / CUDA értelmezés",
            "A gyorsulás ugyanazon komponens és azonos terhelés CPU- és CUDA-futásának teljes idejéből készül. 1× felett a CUDA gyorsabb; 1× alatt a CPU volt gyorsabb. A kis minták és a modell hidegindítása torzíthatják az eredményt.",
        )
        safe_dataframe(speedup_frame, width="stretch", hide_index=True)

    with tabs[6]:
        resources = frame[["Komponens", "Eszköz", "ram_mb", "peak_gpu_memory_mb"]].copy()
        resources.columns = ["Komponens", "Eszköz", "RAM MB", "Peak VRAM MB"]
        resource_usage_chart(resources, key="perf_resources")
        safe_dataframe(resources, width="stretch", hide_index=True)
        with st.expander("Futtatási környezet metaadatai", expanded=False):
            st.json(metadata)

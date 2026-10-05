from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from rag_engine.platform.registry import ExperimentRegistry
from ui.components.charts import delta_bar_chart
from ui.components.common import ROOT
from ui.components.education import empty_state, kpi_cards, note_box, page_intro, section_intro
from ui.components.tables import safe_dataframe


REGISTRY_PATH = ROOT / "artifacts" / "experiments" / "experiments.sqlite3"
EXPORT_PATH = ROOT / "artifacts" / "experiments" / "experiments_export.json"


def _short(value: object, length: int = 12) -> str:
    text = str(value or "")
    return text if len(text) <= length else text[:length] + "…"


def _run_label(run: dict[str, object]) -> str:
    return (
        f"{str(run['created_at'])[:19].replace('T', ' ')} · "
        f"{run['benchmark_type']} · {_short(run['run_id'], 18)}"
    )


def _result_family(row: dict[str, object]) -> str:
    result_type = str(row.get("result_type") or "").lower()
    if "retrieval" in result_type:
        return "retrieval"
    if "rag" in result_type:
        return "rag"
    if "performance" in result_type or "component" in result_type:
        return "performance"
    return "other"


def _variant_label(row: dict[str, object]) -> str:
    family = _result_family(row)
    if family == "retrieval":
        return f"{row.get('chunking') or '—'} · {row.get('retrieval_mode') or row.get('variant') or 'retrieval'} · {row.get('reranker') or 'none'}"
    if family == "rag":
        return f"{row.get('rag_strategy') or row.get('variant') or 'rag'} · {row.get('chunking') or '—'} · {row.get('llm_provider') or '—'}"
    return str(row.get("variant") or row.get("variant_key") or row.get("result_type") or "ismeretlen")


def _safe_json(text: object) -> dict[str, object]:
    try:
        return json.loads(str(text)) if text else {}
    except Exception:
        return {}


def _results_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    drop_cols = [column for column in ["result_id", "extra_json"] if column in frame.columns]
    return frame.drop(columns=drop_cols)


def _compare_metric_table(a: dict[str, object], b: dict[str, object]) -> tuple[pd.DataFrame, str]:
    family_a = _result_family(a)
    family_b = _result_family(b)
    if family_a != family_b:
        return pd.DataFrame(), "mixed"

    if family_a == "retrieval":
        metrics = [
            ("Recall@K", "recall_at_k", "higher"),
            ("Precision@K", "precision_at_k", "higher"),
            ("F1@K", "f1_at_k", "higher"),
            ("Hit Rate@K", "hit_rate_at_k", "higher"),
            ("MRR", "mrr", "higher"),
            ("nDCG@K", "ndcg_at_k", "higher"),
            ("Context Precision@K", "context_precision_at_k", "higher"),
            ("R-Precision", "r_precision", "higher"),
            ("No-hit arány", "no_hit_rate", "lower"),
            ("Latency CV", "latency_cv", "lower"),
            ("Forrásdiverzitás", "source_diversity_at_k", "higher"),
            ("Label coverage", "labeling_coverage", "higher"),
        ]
    elif family_a == "rag":
        metrics = [
            ("Hivatkozási pontosság", "citation_accuracy", "higher"),
            ("Hivatkozási lefedettség", "citation_coverage", "higher"),
            ("Forráslefedettség", "citation_source_coverage", "higher"),
            ("Kulcstény-lefedettség", "key_fact_coverage", "higher"),
            ("Kontextus-kihasználtság", "context_utilization", "higher"),
            ("Fallback arány", "fallback_rate", "lower"),
            ("Latency CV", "latency_cv", "lower"),
            ("Token/s", "mean_tokens_per_second", "higher"),
        ]
    elif family_a == "performance":
        metrics = [
            ("Áteresztőképesség/s", "throughput_per_sec", "higher"),
            ("Mean latency ms", "mean_latency_ms", "lower"),
            ("P95 latency ms", "p95_latency_ms", "lower"),
            ("RAM MB", "ram_mb", "lower"),
            ("Peak VRAM MB", "peak_gpu_memory_mb", "lower"),
        ]
    else:
        return pd.DataFrame(), "other"

    rows: list[dict[str, object]] = []
    for label, key, direction in metrics:
        av = float(a.get(key) or 0.0)
        bv = float(b.get(key) or 0.0)
        delta = bv - av
        effect = delta if direction == "higher" else -delta
        relative = (delta / abs(av) * 100) if abs(av) > 1e-12 else None
        rows.append(
            {
                "Metrika": label,
                "A": av,
                "B": bv,
                "Delta (B-A)": delta,
                "Hatás B vs A": effect,
                "Relatív változás %": relative,
                "Jobb irány": "magasabb" if direction == "higher" else "alacsonyabb",
            }
        )
    return pd.DataFrame(rows), family_a


def _config_diff(a: dict[str, object], b: dict[str, object]) -> pd.DataFrame:
    fields = [
        ("Result type", "result_type"),
        ("Chunking", "chunking"),
        ("Beágyazási modell", "embedding_model"),
        ("Visszakeresés", "retrieval_mode"),
        ("Újrarangsoroló", "reranker"),
        ("RAG stratégia", "rag_strategy"),
        ("Kontextuskeret", "context_budget"),
        ("Beágyazási eszköz", "embedding_device"),
        ("Vector backend", "vector_backend"),
        ("Vector device", "vector_device"),
        ("LLM provider", "llm_provider"),
        ("Kérdések", "questions"),
    ]
    frame = pd.DataFrame(
        [
            {
                "Konfiguráció": label,
                "A": a.get(key),
                "B": b.get(key),
                "Eltér": a.get(key) != b.get(key),
            }
            for label, key in fields
        ]
    )
    return frame


def _history_frame(runs: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Idő": str(run["created_at"])[:19].replace("T", " "),
                "run_id": run["run_id"],
                "Típus": run["benchmark_type"],
                "Státusz": run["status"],
                "Konfigurációhash": _short(run["config_hash"], 14),
                "Kérdések": run["questions"],
                "Időtartam s": round(float(run["duration_ms"] or 0.0) / 1000, 2),
                "Megjegyzés": run["notes"] or "",
            }
            for run in runs
        ]
    )


def render() -> None:
    registry = ExperimentRegistry(REGISTRY_PATH)
    summary = registry.summary()
    all_runs = registry.list_runs(limit=500)

    page_intro(
        "Kísérletek · Benchmark-nyilvántartás",
        "A benchmarkok központi ML/LLM kísérleti nézete: szűrhető futástörténet, részletes futásvizsgáló, A/B konfiguráció- és metrikadelta, valamint reprodukálható export.",
        eyebrow="KÍSÉRLETKÖVETÉS · SZŰRÉS · VIZSGÁLAT · ÖSSZEHASONLÍTÁS · REPRODUKCIÓ",
    )

    kpi_cards(
        [
            ("Benchmark futások", str(summary["runs"]), "Minden benchmark-indítás külön futás."),
            ("Sikeres futások", str(summary["completed"]), f"Hibás: {summary['failed']}"),
            ("Eredménysorok", str(summary["results"]), "Visszakeresési, RAG és teljesítményvariánsok."),
            ("Egyedi konfigurációk", str(summary["unique_configs"]), "Determinista config hash alapján."),
        ],
        columns=4,
    )

    with st.expander("Módszertan · run_id, konfigurációhash és dataset fingerprint", expanded=False):
        st.markdown(
            """
- **run_id**: egy konkrét benchmark invocation egyedi azonosítója.
- **config hash**: normalizált konfiguráció determinisztikus SHA-256 lenyomata.
- **dataset hash**: azt bizonyítja, hogy két futás ugyanazon evaluation dataseten történt-e.
- **A/B delta**: a `Hatás B vs A` oszlop már figyelembe veszi, hogy egy metrikánál a magasabb vagy az alacsonyabb érték a kedvezőbb.
"""
        )

    if not all_runs:
        empty_state(
            "Még nincs kísérleti futás",
            "Futtass visszakeresési, RAG vagy teljes pipeline benchmarkot. A futás automatikusan bekerül az SQLite Experiment Registry-be.",
            hint="Első körben a Teljes pipeline benchmark · Gyors ellenőrzés presetet érdemes használni.",
        )
        return

    section_intro("Futásszűrő", "Szűkítsd a registry-t benchmark típus, státusz vagy run/config azonosító alapján.")
    benchmark_types = sorted({str(run.get("benchmark_type") or "unknown") for run in all_runs})
    statuses = sorted({str(run.get("status") or "unknown") for run in all_runs})
    f1, f2, f3 = st.columns([1, 1, 1.35])
    with f1:
        selected_types = st.multiselect("Benchmark típus", benchmark_types, default=benchmark_types, key="exp_filter_types")
    with f2:
        selected_statuses = st.multiselect("Státusz", statuses, default=statuses, key="exp_filter_status")
    with f3:
        search = st.text_input("Keresés", placeholder="run_id, config hash, megjegyzés…", key="exp_filter_search").strip().lower()

    runs = []
    for run in all_runs:
        if str(run.get("benchmark_type")) not in selected_types or str(run.get("status")) not in selected_statuses:
            continue
        haystack = " ".join(
            str(run.get(key) or "")
            for key in ("run_id", "config_hash", "benchmark_type", "status", "notes", "dataset_hash")
        ).lower()
        if search and search not in haystack:
            continue
        runs.append(run)

    history_tab, detail_tab, compare_tab, export_tab = st.tabs(
        ["Nyilvántartás", "Futásvizsgáló", "A/B összehasonlítás", "Letöltés"]
    )

    with history_tab:
        section_intro("Kísérleti futástörténet", f"{len(runs)} futás felel meg az aktuális szűrőknek.")
        if not runs:
            empty_state("Nincs találat", "Az aktuális szűrőkkel nincs megjeleníthető kísérleti futás.", hint="Tágítsd a típus/státusz szűrőt vagy töröld a keresőkifejezést.")
        else:
            history = _history_frame(runs)
            left, right = st.columns([1.45, 1])
            with left:
                safe_dataframe(history, width="stretch", hide_index=True, height=min(520, 76 + 35 * len(history)))
            with right:
                type_counts = history["Típus"].value_counts().rename_axis("Benchmarktípus").reset_index(name="Futások")
                fig = px.bar(type_counts, x="Futások", y="Benchmarktípus", orientation="h", title="Futások megoszlása benchmarktípus szerint")
                fig.update_layout(height=330, margin=dict(l=16, r=16, t=62, b=32), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, width="stretch", key="experiments_run_types_v22")
                completed_history = history[history["Státusz"] == "completed"].copy()
                if not completed_history.empty:
                    duration = completed_history.groupby("Típus", as_index=False)["Időtartam s"].median().sort_values("Időtartam s")
                    fig2 = px.bar(duration, x="Időtartam s", y="Típus", orientation="h", title="Medián futási idő típusonként")
                    fig2.update_layout(height=300, margin=dict(l=16, r=16, t=62, b=32), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig2, width="stretch", key="experiments_duration_v22")

    with detail_tab:
        section_intro("Futásvizsgáló", "Egy futás konfigurációja, dataset fingerprintje és eredményvariánsai egy nézetben.")
        run_map = {str(run["run_id"]): run for run in runs or all_runs}
        selected_id = st.selectbox("Futás", list(run_map), format_func=lambda rid: _run_label(run_map[rid]), key="experiment_detail_run_v22")
        run = registry.get_run(selected_id) or run_map[selected_id]
        results = registry.get_results(selected_id)
        kpi_cards(
            [
                ("Státusz", str(run["status"]), "A benchmarkfutás állapota."),
                ("Konfigurációhash", _short(run["config_hash"], 14), "Reprodukálható konfigurációazonosító."),
                ("Kérdések", str(run["questions"] or "—"), "Evaluation query-k száma."),
                ("Időtartam", f"{float(run['duration_ms'] or 0.0) / 1000:.2f} s", "Teljes benchmark wall-clock idő."),
            ],
            columns=4,
        )
        if results:
            result_frame = _results_frame(results)
            if "result_type" in result_frame.columns:
                counts = result_frame["result_type"].value_counts().rename_axis("Result type").reset_index(name="Variáns")
                fig = px.bar(counts, x="Result type", y="Variáns", title="Eredményvariánsok összetétele")
                fig.update_layout(height=320, margin=dict(l=16, r=16, t=62, b=36), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, width="stretch", key=f"run_inspector_types_{selected_id}")
            with st.expander("Nyers eredménysorok", expanded=False):
                safe_dataframe(result_frame, width="stretch", hide_index=True)
        else:
            empty_state("Nincs eredménysor", "A run valószínűleg futás közben hibára állt vagy még nem fejeződött be.")
        cfg, fp = st.columns(2)
        with cfg:
            with st.expander("Normalizált konfiguráció", expanded=False):
                st.json(_safe_json(run.get("config_json")))
        with fp:
            with st.expander("Dataset fingerprint", expanded=False):
                st.json({"dataset_path": run.get("dataset_path"), "dataset_hash": run.get("dataset_hash")})

    with compare_tab:
        section_intro(
            "A/B összehasonlítás",
            "Két konkrét benchmarkvariáns közvetlen összehasonlítása. Az eltérésdiagramon a pozitív irány mindig javulást, a negatív regressziót jelent — a metrika természetétől függetlenül.",
        )
        completed = [run for run in all_runs if run.get("status") == "completed" and registry.get_results(str(run["run_id"]))]
        if not completed:
            empty_state("Nincs összehasonlítható futás", "Legalább egy sikeres benchmarkfutás és benne legalább két variáns szükséges az A/B nézethez.")
        else:
            run_ids = [str(run["run_id"]) for run in completed]
            run_lookup = {str(run["run_id"]): run for run in completed}
            ca, cb = st.columns(2)
            with ca:
                run_a = st.selectbox("Futás A", run_ids, format_func=lambda rid: _run_label(run_lookup[rid]), key="compare_run_a_v22")
                results_a = registry.get_results(run_a)
                idx_a = st.selectbox("Variáns A", range(len(results_a)), format_func=lambda idx: _variant_label(results_a[idx]), key="compare_variant_a_v22")
                variant_a = results_a[idx_a]
            with cb:
                default_b = min(1, len(run_ids) - 1)
                run_b = st.selectbox("Futás B", run_ids, index=default_b, format_func=lambda rid: _run_label(run_lookup[rid]), key="compare_run_b_v22")
                results_b = registry.get_results(run_b)
                idx_b = st.selectbox("Variáns B", range(len(results_b)), index=min(1, len(results_b) - 1), format_func=lambda idx: _variant_label(results_b[idx]), key="compare_variant_b_v22")
                variant_b = results_b[idx_b]

            diff = _config_diff(variant_a, variant_b)
            changed = diff[diff["Eltér"]].copy()
            with st.expander(f"Konfigurációs eltérés · {len(changed)} eltérés", expanded=False):
                safe_dataframe(changed if not changed.empty else diff.head(0), width="stretch", hide_index=True)

            metric_frame, family = _compare_metric_table(variant_a, variant_b)
            if family == "mixed":
                st.warning("Eltérő eredménytípusokat választottál. Válassz két retrieval, két RAG vagy két performance variánst a quality delta megjelenítéséhez.")
            elif metric_frame.empty:
                empty_state("Nincs közös metrikakészlet", "A kiválasztott eredménytípushoz még nincs definiált A/B scorecard.")
            else:
                improved = int((metric_frame["Hatás B vs A"] > 0).sum())
                regressed = int((metric_frame["Hatás B vs A"] < 0).sum())
                unchanged = int((metric_frame["Hatás B vs A"].abs() < 1e-12).sum())
                kpi_cards(
                    [
                        ("Javuló metrikák", str(improved), "B kedvezőbb A-nál a metrika helyes irányát figyelembe véve."),
                        ("Romló metrikák", str(regressed), "B regressziót mutat A-hoz képest."),
                        ("Változatlan", str(unchanged), "A mért különbség gyakorlatilag nulla."),
                        ("Legnagyobb hatás", f"{metric_frame['Hatás B vs A'].abs().max():.3f}", "A legnagyobb abszolút irányhelyes delta."),
                    ],
                    columns=4,
                )
                delta_bar_chart(
                    metric_frame,
                    metric_col="Metrika",
                    delta_col="Hatás B vs A",
                    key="experiment_ab_delta_v22",
                    title="A/B eltérésdiagram · pozitív = javulás",
                )
                with st.expander("Metrikatábla · A / B / delta", expanded=False):
                    safe_dataframe(metric_frame, width="stretch", hide_index=True)

            latency_a = float(variant_a.get("mean_latency_ms") or 0.0)
            latency_b = float(variant_b.get("mean_latency_ms") or 0.0)
            p95_a = float(variant_a.get("p95_latency_ms") or 0.0)
            p95_b = float(variant_b.get("p95_latency_ms") or 0.0)
            kpi_cards(
                [
                    ("A mean latency", f"{latency_a:.1f} ms", _variant_label(variant_a)),
                    ("B mean latency", f"{latency_b:.1f} ms", f"Δ {latency_b - latency_a:+.1f} ms"),
                    ("A P95", f"{p95_a:.1f} ms", "Kiinduló tail latency."),
                    ("B P95", f"{p95_b:.1f} ms", f"Δ {p95_b - p95_a:+.1f} ms"),
                ],
                columns=4,
            )
            note_box(
                "A/B értelmezés",
                "A delta diagram a metrika kedvező irányát normalizálja, de nem választ automatikus győztest. A minőség, a késleltetés és az erőforrásigény együtt értelmezendő.",
            )

    with export_tab:
        section_intro("Registry exportálása", "A teljes futástörténet JSON-ba exportálható, és később benchmarkjelentés vagy CI artifact bemeneteként használható.")
        export_path = registry.export_json(EXPORT_PATH)
        st.code(str(REGISTRY_PATH), language="text")
        st.download_button(
            "Registry JSON export letöltése",
            data=export_path.read_bytes(),
            file_name=export_path.name,
            mime="application/json",
            width="stretch",
        )

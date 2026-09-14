from __future__ import annotations

import time
import math
from pathlib import Path
from typing import Callable, Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from prompt_benchmark.paths import PATHS
from prompt_benchmark.benchmark.runner import run_strategy
from prompt_benchmark.constants import LABELS
from prompt_benchmark.data.benchmark_suites import (
    normalize_benchmark_frame,
    prepare_hf_support_router_suite,
    profile_dataset,
    representative_stratified_subset,
)
from prompt_benchmark.evaluation.metrics import classification_metrics
from prompt_benchmark.evaluation.bootstrap import bootstrap_macro_f1_ci
from prompt_benchmark.prompts import get_strategy, list_custom_prompts, load_custom_prompt
from prompt_benchmark.utils.pricing import get_provider_pricing
from prompt_benchmark.ui.run_history import make_run_id, initialize_run, finalize_run, mark_failed, publish_latest


TEXT = {
    "hu": {
        "title": "Kontrollált benchmark — élő futásvezérlő",
        "intro": "A benchmark itt ténylegesen requestenként fut. Futás közben látod az aktuális stratégiát, mintát, progresszt, tokeneket, latencyt és részleges minőségi mutatókat.",
        "dataset": "1. Benchmark dataset",
        "challenge": "🎯 Offline Prompt Challenge Set — ajánlott prompttechnikák összehasonlítására",
        "external": "🌐 Külső HF Support Router — generalizációs ellenőrzés",
        "development": "🧪 Development split — promptfejlesztéshez, nem final riporthoz",
        "custom": "📁 Saját CSV feltöltése",
        "challenge_help": "Szintetikus, de szándékosan nehéz: ambiguity, multi-intent, zaj, hosszú kontextus, prompt injection, resolved history. Arra készült, hogy a prompttechnikák robusztussága különváljon.",
        "external_help": "Független Hugging Face test split. Jó generalizációs kontroll, de maga a forrás is szintetikus és a dataset készítői kiszűrték a noisy/ambiguous mintákat.",
        "development_help": "Ezen szabad promptot iterálni. A final benchmarkot ne erre optimalizáld.",
        "custom_help": "CSV oszlopok: text,label. Engedélyezett label-ek: api, billing, cancellation, complaint, technical, upgrade.",
        "download": "HF suite letöltése és előkészítése",
        "upload": "CSV feltöltése",
        "dataset_missing": "A kiválasztott suite még nincs előkészítve.",
        "rows": "Minták",
        "classes": "Osztályok",
        "avg_words": "Átlag szó/minta",
        "hard_share": "Hard esetek",
        "balance": "Min/max class",
        "preview": "Dataset előnézet",
        "case_mix": "Esettípus-megoszlás",
        "case_filter": "Esettípus szűrő",
        "difficulty_filter": "Nehézség szűrő",
        "strategy": "2. Promptstratégiák",
        "select": "Promptok / stratégiák",
        "limit_on": "Mintaszám limit",
        "limit": "Minták / stratégia",
        "force": "Újrafuttatás / cache felülírás",
        "plan": "3. Futási terv",
        "provider": "Provider / modell",
        "samples": "Összes értékelt minta",
        "calls": "Becsült LLM/API hívás",
        "prompt_tokens": "Durva prompt-token budget",
        "cloud_warning": "Cloud API-nál először 6–24 mintával próbáld ki. A P14 branch+vote 3 hívást használ mintánként.",
        "start": "▶ Benchmark indítása",
        "running": "Benchmark fut…",
        "current_strategy": "Aktuális stratégia",
        "current_sample": "Aktuális minta",
        "overall": "Teljes progress",
        "strategy_progress": "Stratégia progress",
        "live_accuracy": "Live accuracy",
        "live_f1": "Live Macro F1",
        "tokens": "Felhasznált token",
        "latency": "Összes latency",
        "last_request": "Legutóbbi request",
        "partial": "Elkészült stratégiák — élő leaderboard",
        "done": "Benchmark kész.",
        "failed": "Benchmark hiba",
        "no_custom": "Nincs mentett custom prompt.",
        "external_note": "A HF Support-Ticket-Router-12K-Cleaned szintén szintetikus; ezt external/generalization kontrollként kezeljük, nem 'real customer' adatként.",
        "custom_saved": "A feltöltött benchmark ideiglenesen mentve.",
        "source": "Forrás",
        "run_id": "Futásazonosító",
        "history_saved": "A futás bekerült a benchmark historyba.",
        "post_dashboard": "Azonnali eredménydashboard",
        "validation_ready": "Az Output validáció már használható ehhez a futáshoz.",
    },
    "en": {
        "title": "Controlled benchmark — live run controller",
        "intro": "The benchmark executes request by request. While it runs you can see the active strategy, sample, progress, tokens, latency and partial quality metrics.",
        "dataset": "1. Benchmark dataset",
        "challenge": "🎯 Offline Prompt Challenge Set — recommended for prompt-technique comparison",
        "external": "🌐 External HF Support Router — generalization check",
        "development": "🧪 Development split — for prompt iteration, not final reporting",
        "custom": "📁 Upload custom CSV",
        "challenge_help": "Synthetic but deliberately difficult: ambiguity, multi-intent, noise, long context, prompt injection and resolved history. It is designed to separate prompt robustness.",
        "external_help": "Independent Hugging Face test split. Useful as a generalization control, but the source itself is synthetic and its authors filtered noisy/ambiguous examples.",
        "development_help": "Use this for prompt iteration. Do not optimize the final benchmark on it.",
        "custom_help": "CSV columns: text,label. Allowed labels: api, billing, cancellation, complaint, technical, upgrade.",
        "download": "Download and prepare HF suite",
        "upload": "Upload CSV",
        "dataset_missing": "The selected suite has not been prepared yet.",
        "rows": "Rows",
        "classes": "Classes",
        "avg_words": "Avg words/sample",
        "hard_share": "Hard cases",
        "balance": "Min/max class",
        "preview": "Dataset preview",
        "case_mix": "Case-type mix",
        "case_filter": "Case-type filter",
        "difficulty_filter": "Difficulty filter",
        "strategy": "2. Prompt strategies",
        "select": "Prompts / strategies",
        "limit_on": "Limit sample count",
        "limit": "Samples / strategy",
        "force": "Rerun / overwrite cache",
        "plan": "3. Run plan",
        "provider": "Provider / model",
        "samples": "Total evaluated samples",
        "calls": "Estimated LLM/API calls",
        "prompt_tokens": "Rough prompt-token budget",
        "cloud_warning": "For cloud APIs start with 6–24 samples. P14 branch+vote uses 3 calls per sample.",
        "start": "▶ Start benchmark",
        "running": "Benchmark running…",
        "current_strategy": "Current strategy",
        "current_sample": "Current sample",
        "overall": "Overall progress",
        "strategy_progress": "Strategy progress",
        "live_accuracy": "Live accuracy",
        "live_f1": "Live Macro F1",
        "tokens": "Tokens used",
        "latency": "Total latency",
        "last_request": "Latest request",
        "partial": "Completed strategies — live leaderboard",
        "done": "Benchmark complete.",
        "failed": "Benchmark error",
        "no_custom": "No saved custom prompts.",
        "external_note": "HF Support-Ticket-Router-12K-Cleaned is also synthetic; treat it as an external/generalization control, not as real-customer data.",
        "custom_saved": "Uploaded benchmark saved temporarily.",
        "source": "Source",
        "run_id": "Run ID",
        "history_saved": "The run was persisted to benchmark history.",
        "post_dashboard": "Immediate result dashboard",
        "validation_ready": "Output Validation is now available for this run.",
    },
}


def _txt(language: str, key: str) -> str:
    return TEXT[language][key]


def _estimate_strategy_calls_and_tokens(strategy: Any, sample_text: str, sample_count: int) -> tuple[int, int]:
    if hasattr(strategy, "build_branches"):
        payloads = strategy.build_branches(sample_text)
    else:
        payloads = [strategy.build(sample_text)]
    calls = len(payloads) * sample_count
    approx_tokens_one = sum(max(1, int((len(p.instructions or "") + len(p.input_text)) / 4)) for p in payloads)
    return calls, approx_tokens_one * sample_count


def _metric_row(strategy_name: str, display: str, frame: pd.DataFrame) -> dict[str, Any]:
    metrics = classification_metrics(frame)
    ci_low, ci_high = bootstrap_macro_f1_ci(frame, iterations=400, random_seed=42)
    row: dict[str, Any] = {
        "strategy": strategy_name,
        "strategy_title": display,
        **metrics,
        "macro_f1_ci_low": ci_low,
        "macro_f1_ci_high": ci_high,
        "class_coverage": int(frame["true_label"].astype(str).nunique()),
        "case_type_coverage": int(frame["case_type"].fillna("").astype(str).replace("", pd.NA).dropna().nunique()) if "case_type" in frame else 0,
    }
    if "difficulty" in frame:
        for difficulty in ("easy", "medium", "hard"):
            group = frame[frame["difficulty"].astype(str).str.lower() == difficulty]
            if not group.empty:
                gm = classification_metrics(group)
                row[f"{difficulty}_accuracy"] = gm["accuracy"]
                row[f"{difficulty}_macro_f1"] = gm["macro_f1"]
                row[f"{difficulty}_samples"] = int(len(group))
    return row




def _performance_figure(summary: pd.DataFrame, language: str) -> go.Figure:
    work = summary.copy().sort_values("macro_f1", ascending=False)
    if "accuracy_ci_low" not in work:
        work["accuracy_ci_low"] = work["accuracy"]
        work["accuracy_ci_high"] = work["accuracy"]
    work["acc_err_plus"] = (work["accuracy_ci_high"] - work["accuracy"]).clip(lower=0)
    work["acc_err_minus"] = (work["accuracy"] - work["accuracy_ci_low"]).clip(lower=0)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Accuracy", x=work["strategy_title"], y=work["accuracy"],
        error_y=dict(type="data", array=work["acc_err_plus"], arrayminus=work["acc_err_minus"]),
        text=[f"n={int(v)}" for v in work.get("requests", pd.Series([0] * len(work)))],
        textposition="outside",
        hovertemplate="%{x}<br>Accuracy=%{y:.3f}<br>%{text}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        name="Macro F1", x=work["strategy_title"], y=work["macro_f1"],
        mode="markers+lines", marker=dict(size=10),
        hovertemplate="%{x}<br>Macro F1=%{y:.3f}<extra></extra>",
    ))
    observed_min = float(min(work["accuracy"].min(), work["macro_f1"].min())) if len(work) else 0.0
    y0 = max(0.0, observed_min - 0.12)
    fig.update_layout(
        title="Teljesítmény + Accuracy 95% CI" if language == "hu" else "Performance + Accuracy 95% CI",
        yaxis_title="Score", yaxis_range=[y0, 1.05], xaxis_title="", barmode="group",
        legend_orientation="h", legend_y=1.12,
    )
    return fig


def _efficiency_figure(summary: pd.DataFrame, language: str) -> go.Figure:
    hover = [c for c in ["accuracy", "balanced_accuracy", "matthews_corrcoef", "p95_latency_seconds", "requests", "hard_macro_f1"] if c in summary]
    fig = px.scatter(
        summary, x="mean_total_tokens", y="macro_f1", size="p95_latency_seconds",
        color="strategy_title", hover_name="strategy_title", hover_data=hover,
        title="Minőség vs token vs latency" if language == "hu" else "Quality vs tokens vs latency",
        labels={"mean_total_tokens": "Token / request", "macro_f1": "Macro F1", "p95_latency_seconds": "P95 latency"},
    )
    if len(summary):
        ymin = max(0.0, float(summary["macro_f1"].min()) - 0.10)
        fig.update_yaxes(range=[ymin, 1.03])
    return fig


def _difficulty_figure(summary: pd.DataFrame, language: str) -> go.Figure | None:
    cols = [c for c in ["easy_macro_f1", "medium_macro_f1", "hard_macro_f1"] if c in summary]
    if not cols:
        return None
    work = summary[["strategy_title", *cols]].melt(id_vars="strategy_title", var_name="difficulty", value_name="macro_f1")
    work["difficulty"] = work["difficulty"].str.replace("_macro_f1", "", regex=False)
    return px.bar(
        work, x="strategy_title", y="macro_f1", color="difficulty", barmode="group",
        range_y=[0, 1.03], title="Macro F1 nehézségi szintenként" if language == "hu" else "Macro F1 by difficulty",
    )


def _case_heatmap(history_root: Path, summary: pd.DataFrame, language: str) -> go.Figure | None:
    rows: list[dict[str, Any]] = []
    title_map = dict(zip(summary["strategy"].astype(str), summary["strategy_title"].astype(str))) if len(summary) else {}
    for raw_path in (history_root / "raw").glob("*.csv"):
        frame = pd.read_csv(raw_path)
        if "case_type" not in frame or frame["case_type"].fillna("").astype(str).eq("").all():
            continue
        for case_type, group in frame.groupby("case_type"):
            if not str(case_type):
                continue
            rows.append({
                "strategy_title": title_map.get(raw_path.stem, raw_path.stem),
                "case_type": str(case_type),
                "accuracy": float(group["correct"].astype(bool).mean()),
            })
    if not rows:
        return None
    pivot = pd.DataFrame(rows).pivot(index="strategy_title", columns="case_type", values="accuracy")
    return px.imshow(
        pivot, text_auto=".2f", zmin=0, zmax=1, aspect="auto",
        title="Pontosság scenario típusonként" if language == "hu" else "Accuracy by scenario type",
        labels=dict(color="Accuracy"),
    )


def _mcnemar_exact_p(fixed: int, regressed: int) -> float:
    discordant = fixed + regressed
    if discordant == 0:
        return 1.0
    k = min(fixed, regressed)
    tail = sum(math.comb(discordant, i) for i in range(k + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def _paired_vs_baseline(history_root: Path, summary: pd.DataFrame) -> pd.DataFrame:
    baseline_path = history_root / "raw" / "p0_zero_shot.csv"
    if not baseline_path.exists():
        return pd.DataFrame()
    baseline = pd.read_csv(baseline_path)[["sample_id", "correct"]].rename(columns={"correct": "baseline_correct"})
    title_map = dict(zip(summary["strategy"].astype(str), summary["strategy_title"].astype(str))) if "strategy_title" in summary else {}
    rows: list[dict[str, Any]] = []
    for raw_path in sorted((history_root / "raw").glob("*.csv")):
        if raw_path.stem == "p0_zero_shot":
            continue
        current = pd.read_csv(raw_path)[["sample_id", "correct"]].rename(columns={"correct": "strategy_correct"})
        merged = baseline.merge(current, on="sample_id", how="inner")
        if merged.empty:
            continue
        b = merged["baseline_correct"].astype(bool)
        c = merged["strategy_correct"].astype(bool)
        fixed = int((~b & c).sum())
        regressed = int((b & ~c).sum())
        rows.append({
            "strategy": raw_path.stem,
            "strategy_title": title_map.get(raw_path.stem, raw_path.stem),
            "samples": len(merged),
            "fixed_vs_p0": fixed,
            "regressed_vs_p0": regressed,
            "net_fixed": fixed - regressed,
            "paired_accuracy_delta_pp": (fixed - regressed) / len(merged) * 100.0,
            "mcnemar_exact_p": _mcnemar_exact_p(fixed, regressed),
        })
    return pd.DataFrame(rows).sort_values(["net_fixed", "fixed_vs_p0"], ascending=False) if rows else pd.DataFrame()


def _sample_quality_message(n: int, accuracy: float, ci_low: float, ci_high: float, language: str) -> str:
    if n < 60:
        prefix = "⚠ PILOT"
    elif n < 120:
        prefix = "ℹ QUICK"
    else:
        prefix = "✅ BENCHMARK"
    if language == "hu":
        return f"{prefix}: n={n}. Accuracy={accuracy:.3f}, Wilson 95% CI=[{ci_low:.3f}, {ci_high:.3f}]. Kis mintán az 1.000 nem bizonyít tökéletes általánosítást."
    return f"{prefix}: n={n}. Accuracy={accuracy:.3f}, Wilson 95% CI=[{ci_low:.3f}, {ci_high:.3f}]. On a small sample, 1.000 does not prove perfect generalization."


def render_benchmark_panel(
    *,
    language: str,
    provider: str,
    model: str,
    strategies: list[str],
    strategy_title: Callable[[str], str],
    build_client: Callable[[], Any],
    pricing_config: dict[str, Any],
) -> None:
    t = lambda key: _txt(language, key)
    st.subheader(t("title"))
    st.write(t("intro"))

    st.markdown(f"### {t('dataset')}")
    source_labels = [t("challenge"), t("external"), t("development"), t("custom")]
    source = st.radio(t("source"), source_labels, horizontal=False, key=f"{language}_bench_suite_source")
    suite_key = "challenge"
    examples_path: str | Path = PATHS.prompt_examples / "few_shot_examples.json"
    data: pd.DataFrame | None = None

    if source == t("challenge"):
        suite_key = "challenge"
        st.info(t("challenge_help"))
        path = PATHS.processed_data / "benchmark.csv"
        if path.exists():
            data = pd.read_csv(path)
    elif source == t("development"):
        suite_key = "development"
        st.warning(t("development_help"))
        path = PATHS.processed_data / "development.csv"
        if path.exists():
            data = pd.read_csv(path)
    elif source == t("external"):
        suite_key = "hf_external"
        st.info(t("external_help"))
        st.caption(t("external_note"))
        path = PATHS.benchmark_suites / "hf_support_router_300.csv"
        examples_path = PATHS.benchmark_suites / "hf_support_router_few_shot.json"
        if not path.exists():
            if st.button(t("download"), key=f"{language}_prepare_hf_suite"):
                with st.status(t("download"), expanded=True) as status:
                    try:
                        out, ex = prepare_hf_support_router_suite()
                        status.write(str(out))
                        status.write(str(ex))
                        status.update(label=t("done"), state="complete")
                        st.rerun()
                    except Exception as exc:
                        status.update(label=f"{t('failed')}: {exc}", state="error")
        if path.exists():
            data = pd.read_csv(path)
        else:
            st.warning(t("dataset_missing"))
    else:
        suite_key = "custom_upload"
        st.info(t("custom_help"))
        uploaded = st.file_uploader(t("upload"), type=["csv"], key=f"{language}_benchmark_csv")
        if uploaded is not None:
            try:
                raw = pd.read_csv(uploaded)
                data = normalize_benchmark_frame(raw, prefix="upload")
                out = PATHS.benchmark_suites / "custom_uploaded.csv"
                out.parent.mkdir(parents=True, exist_ok=True)
                data.to_csv(out, index=False)
                st.success(t("custom_saved"))
            except Exception as exc:
                st.error(f"{t('failed')}: {type(exc).__name__}: {exc}")

    if data is None or data.empty:
        return

    # Dataset observability before the user spends API calls.
    try:
        profile = profile_dataset(data, name=suite_key)
        cols = st.columns(5)
        cols[0].metric(t("rows"), f"{profile.rows:,}")
        cols[1].metric(t("classes"), profile.labels)
        cols[2].metric(t("avg_words"), f"{profile.mean_words:.1f}")
        cols[3].metric(t("hard_share"), "N/A" if profile.hard_share is None else f"{profile.hard_share:.0%}")
        cols[4].metric(t("balance"), f"{profile.min_class_size}/{profile.max_class_size}")
    except Exception as exc:
        st.warning(f"Dataset profile: {exc}")

    a, b = st.columns([1.2, 1])
    with a:
        with st.expander(t("preview"), expanded=True):
            preview_cols = [c for c in ["sample_id", "text", "true_label", "case_type", "difficulty"] if c in data]
            st.dataframe(data[preview_cols].head(12), use_container_width=True, hide_index=True)
    with b:
        if "true_label" in data:
            counts = data["true_label"].value_counts().rename_axis("label").reset_index(name="count")
            st.plotly_chart(px.bar(counts, x="label", y="count", title=t("balance")), use_container_width=True)
        if "case_type" in data and data["case_type"].fillna("").astype(str).str.len().gt(0).any():
            cases = data["case_type"].value_counts().rename_axis("case_type").reset_index(name="count")
            st.plotly_chart(px.bar(cases, x="case_type", y="count", title=t("case_mix")), use_container_width=True)

    # Optional robustness slices. The user can benchmark only hard / injection /
    # multi-intent cases without creating a second dataset file.
    filtered = data.copy()
    if "case_type" in filtered and filtered["case_type"].fillna("").astype(str).str.len().gt(0).any():
        all_cases = sorted(x for x in filtered["case_type"].fillna("").astype(str).unique() if x)
        selected_cases = st.multiselect(t("case_filter"), all_cases, default=all_cases, key=f"{language}_suite_case_filter")
        if selected_cases:
            filtered = filtered[filtered["case_type"].astype(str).isin(selected_cases)]
    if "difficulty" in filtered and filtered["difficulty"].fillna("").astype(str).str.len().gt(0).any():
        all_diff = sorted(x for x in filtered["difficulty"].fillna("").astype(str).unique() if x)
        selected_diff = st.multiselect(t("difficulty_filter"), all_diff, default=all_diff, key=f"{language}_suite_difficulty_filter")
        if selected_diff:
            filtered = filtered[filtered["difficulty"].astype(str).isin(selected_diff)]
    data = filtered.reset_index(drop=True)
    if data.empty:
        st.warning(t("dataset_missing"))
        return

    st.markdown(f"### {t('strategy')}")
    built_options = {strategy_title(s): ("builtin", s) for s in strategies}
    saved = {p.stem: p for p in list_custom_prompts()}
    custom_options = {f"CUSTOM · {name}": ("custom", path) for name, path in saved.items()}
    options = {**built_options, **custom_options}
    default_names = [strategy_title(s) for s in ["p0_zero_shot", "p3_few_shot", "p7_structured_output", "p11_delimited_data", "p16_full_advanced_template"] if s in strategies]
    chosen = st.multiselect(t("select"), list(options), default=[x for x in default_names if x in options], key=f"{language}_live_bench_strategies")

    max_n = max(1, len(data))
    profiles = {
        ("Smoke · 12" if language == "hu" else "Smoke · 12"): min(12, max_n),
        ("Pilot · 36" if language == "hu" else "Pilot · 36"): min(36, max_n),
        ("Standard · 120" if language == "hu" else "Standard · 120"): min(120, max_n),
        ("Strong · 300" if language == "hu" else "Strong · 300"): min(300, max_n),
        ("Full dataset" if language == "hu" else "Full dataset"): max_n,
        ("Egyedi" if language == "hu" else "Custom"): None,
    }
    default_profile = "Pilot · 36"
    profile_name = st.selectbox(
        "Futási profil" if language == "hu" else "Run profile",
        list(profiles),
        index=list(profiles).index(default_profile),
        key=f"{language}_benchmark_run_profile_v4",
        help=(
            "Minden profil reprezentatív, stratifikált subsetet választ; nem egyszerűen az első N sort. Smoke/Pilot csak UX/API teszt, final következtetéshez legalább Standard/Strong ajánlott."
            if language == "hu" else
            "Every profile selects a representative stratified subset instead of the first N rows. Smoke/Pilot are for UX/API checks; use at least Standard/Strong for final conclusions."
        ),
    )
    if profiles[profile_name] is None:
        n_samples = int(st.number_input(t("limit"), 6, max_n, min(120, max_n), 1, key=f"{language}_live_limit_v4"))
    else:
        n_samples = int(profiles[profile_name])
    force = st.checkbox(t("force"), True, key=f"{language}_live_force")

    # Never benchmark ``head(n)``. A deterministic label+scenario stratified pilot
    # prevents a tiny easy subset from looking like a perfect benchmark.
    run_data = representative_stratified_subset(data, n_samples, random_seed=42)
    coverage_cols = st.columns(5)
    coverage_cols[0].metric("Pilot N" if language == "hu" else "Pilot N", f"{len(run_data):,}")
    coverage_cols[1].metric("Class coverage" if language == "en" else "Class lefedettség", f"{run_data['true_label'].nunique()}/6")
    case_cov = run_data['case_type'].astype(str).replace('', pd.NA).dropna().nunique() if 'case_type' in run_data else 0
    coverage_cols[2].metric("Scenario coverage" if language == "en" else "Scenario lefedettség", f"{case_cov}")
    hard_share = run_data['difficulty'].astype(str).str.lower().eq('hard').mean() if 'difficulty' in run_data else float('nan')
    coverage_cols[3].metric("Hard share" if language == "en" else "Hard arány", "—" if pd.isna(hard_share) else f"{hard_share:.0%}")
    coverage_cols[4].metric("Full holdout", f"{len(data):,}")
    if n_samples < 60:
        st.warning(
            "⚠ Ez pilot futás. 1.00 Accuracy/F1 kis mintán teljesen lehetséges; ne tekintsd final modellminőségnek. A dashboard Wilson 95% CI-t és mintaszámot is mutat."
            if language == "hu" else
            "⚠ This is a pilot run. Accuracy/F1=1.00 is entirely possible on a small sample; do not treat it as final model quality. The dashboard also shows Wilson 95% CI and sample count."
        )

    selected_objects: list[tuple[str, Any]] = []
    for display in chosen:
        kind, value = options[display]
        if kind == "builtin":
            selected_objects.append((display, get_strategy(value, examples_path=examples_path)))
        else:
            selected_objects.append((display, load_custom_prompt(value)))

    st.markdown(f"### {t('plan')}")
    total_calls = 0
    rough_prompt_tokens = 0
    sample_text = str(run_data.iloc[0]["text"]) if len(run_data) else ""
    for _, strategy in selected_objects:
        calls, toks = _estimate_strategy_calls_and_tokens(strategy, sample_text, n_samples)
        total_calls += calls
        rough_prompt_tokens += toks
    pcols = st.columns(4)
    pcols[0].metric(t("provider"), f"{provider} / {model}")
    pcols[1].metric(t("samples"), f"{n_samples * len(selected_objects):,}")
    pcols[2].metric(t("calls"), f"{total_calls:,}")
    pcols[3].metric(t("prompt_tokens"), f"~{rough_prompt_tokens:,}")
    if provider not in {"mock", "ollama"}:
        st.warning(t("cloud_warning"))

    if not selected_objects:
        return

    if st.button(t("start"), type="primary", key=f"{language}_live_bench_start"):
        try:
            client = build_client()
            input_price, output_price = get_provider_pricing(pricing_config, provider)
            run_start = time.perf_counter()
            run_id = make_run_id(provider, suite_key)
            selected_strategy_names = [getattr(obj, "name", display) for display, obj in selected_objects]
            manifest = {
                "provider": provider,
                "model": model,
                "suite": suite_key,
                "language": language,
                "sample_count_per_strategy": n_samples,
                "selected_strategies": selected_strategy_names,
                "case_filter": selected_cases if "selected_cases" in locals() else [],
                "difficulty_filter": selected_diff if "selected_diff" in locals() else [],
                "estimated_calls": total_calls,
                "estimated_prompt_tokens": rough_prompt_tokens,
            }
            history_root = initialize_run(run_id, manifest, run_data.copy())
            st.session_state["active_benchmark_run_id"] = run_id
            st.caption(f"{t('run_id')}: `{run_id}`")
            overall_bar = st.progress(0.0, text=t("overall"))
            strategy_bar = st.progress(0.0, text=t("strategy_progress"))
            status = st.status(t("running"), expanded=True)
            strategy_ph = st.empty()
            sample_ph = st.empty()
            metrics_ph = st.empty()
            last_ph = st.empty()
            leaderboard_ph = st.empty()

            completed_rows: list[dict[str, Any]] = []
            total_strategies = len(selected_objects)

            for s_idx, (display, strategy) in enumerate(selected_objects):
                strategy_ph.info(f"{t('current_strategy')}: {display} ({s_idx + 1}/{total_strategies})")
                partial_rows: list[dict[str, Any]] = []
                strategy_started = time.perf_counter()

                def on_progress(event: dict[str, Any]) -> None:
                    if event.get("event") != "sample":
                        return
                    row = event["row"]
                    partial_rows.append(row)
                    current = int(event.get("completed", len(partial_rows)))
                    total = max(1, int(event.get("total", n_samples)))
                    strategy_bar.progress(min(1.0, current / total), text=f"{t('strategy_progress')}: {current}/{total}")
                    overall_done = s_idx * n_samples + min(current, n_samples)
                    overall_total = max(1, total_strategies * n_samples)
                    overall_bar.progress(min(1.0, overall_done / overall_total), text=f"{t('overall')}: {overall_done}/{overall_total}")
                    sample_ph.caption(f"{t('current_sample')}: {row.get('sample_id')} · true={row.get('true_label')} · pred={row.get('predicted_label')} · case={row.get('case_type') or '-'}")
                    pf = pd.DataFrame(partial_rows)
                    if len(pf):
                        m = classification_metrics(pf)
                        with metrics_ph.container():
                            mc = st.columns(6)
                            mc[0].metric(t("live_accuracy"), f"{m['accuracy']:.3f}", f"95% CI {m['accuracy_ci_low']:.2f}–{m['accuracy_ci_high']:.2f}")
                            mc[1].metric(t("live_f1"), f"{m['macro_f1']:.3f}")
                            mc[2].metric("Balanced Acc.", f"{m['balanced_accuracy']:.3f}")
                            mc[3].metric(t("tokens"), f"{m['total_tokens']:.0f}")
                            mc[4].metric(t("latency"), f"{m['total_latency_seconds']:.2f}s")
                            mc[5].metric("n", f"{len(pf)}/{n_samples}")
                            st.caption(_sample_quality_message(len(pf), m['accuracy'], m['accuracy_ci_low'], m['accuracy_ci_high'], language))
                    with last_ph.container():
                        st.markdown(f"#### {t('last_request')}")
                        st.dataframe(pd.DataFrame([{
                            "sample_id": row.get("sample_id"),
                            "true": row.get("true_label"),
                            "pred": row.get("predicted_label"),
                            "correct": row.get("correct"),
                            "tokens": row.get("total_tokens"),
                            "latency_s": row.get("latency_seconds"),
                            "valid": row.get("valid_output"),
                            "error": row.get("error"),
                        }]), hide_index=True, use_container_width=True)

                output_path = history_root / "raw" / f"{strategy.name}.csv"
                result = run_strategy(
                    run_data,
                    strategy,
                    client,
                    output_path,
                    input_price,
                    output_price,
                    limit=None,
                    force=force,
                    progress_callback=on_progress,
                )
                row = _metric_row(strategy.name, display, result)
                row["elapsed_wall_seconds"] = time.perf_counter() - strategy_started
                completed_rows.append(row)
                live = pd.DataFrame(completed_rows)
                with leaderboard_ph.container():
                    st.markdown(f"#### {t('partial')}")
                    show = ["strategy_title", "accuracy", "macro_f1", "output_contract_valid_rate", "mean_total_tokens", "total_tokens", "p95_latency_seconds", "benchmark_cost_usd", "elapsed_wall_seconds"]
                    st.dataframe(live[[c for c in show if c in live]], use_container_width=True, hide_index=True)
                    c1, c2 = st.columns(2)
                    c1.plotly_chart(_performance_figure(live, language), use_container_width=True, key=f"live_f1_{language}_{s_idx}")
                    c2.plotly_chart(_efficiency_figure(live, language), use_container_width=True, key=f"live_tokens_{language}_{s_idx}")

            elapsed = time.perf_counter() - run_start
            overall_bar.progress(1.0, text=f"{t('done')} {elapsed:.1f}s")
            strategy_bar.progress(1.0, text=t("done"))
            status.update(label=f"{t('done')} ({elapsed:.1f}s)", state="complete", expanded=False)
            summary = pd.DataFrame(completed_rows)
            case_rows: list[dict[str, Any]] = []
            diff_rows: list[dict[str, Any]] = []
            for raw_path in (history_root / "raw").glob("*.csv"):
                raw_frame = pd.read_csv(raw_path)
                strategy_name = raw_path.stem
                if "case_type" in raw_frame:
                    for case_type, group in raw_frame.groupby("case_type", dropna=False):
                        if str(case_type):
                            case_rows.append({"strategy": strategy_name, "case_type": case_type, "accuracy": float(group["correct"].astype(bool).mean()), "samples": len(group)})
                if "difficulty" in raw_frame:
                    for difficulty, group in raw_frame.groupby("difficulty", dropna=False):
                        if str(difficulty):
                            diff_rows.append({"strategy": strategy_name, "difficulty": difficulty, "accuracy": float(group["correct"].astype(bool).mean()), "samples": len(group)})
            if case_rows:
                pd.DataFrame(case_rows).to_csv(history_root / "case_type_summary.csv", index=False)
            if diff_rows:
                pd.DataFrame(diff_rows).to_csv(history_root / "difficulty_summary.csv", index=False)
            paired = _paired_vs_baseline(history_root, summary)
            if not paired.empty:
                paired.to_csv(history_root / "paired_vs_p0.csv", index=False)
            finalize_run(run_id, summary, {
                "elapsed_wall_seconds": elapsed,
                "rows_in_dataset_snapshot": int(min(n_samples, len(data))),
            })
            publish_latest(run_id, provider)
            st.session_state[f"last_live_benchmark_{language}"] = summary
            st.session_state["latest_benchmark_run_id"] = run_id
            st.session_state["benchmark_completed"] = True
            st.success(f"{t('done')} {t('history_saved')}")
            st.info(t("validation_ready"))
            st.markdown(f"### {t('post_dashboard')}")
            d1, d2, d3, d4, d5 = st.columns(5)
            best = summary.sort_values("macro_f1", ascending=False).iloc[0]
            d1.metric("Best Macro F1", f"{best['macro_f1']:.4f}", str(best.get('strategy_title', best.get('strategy', ''))))
            d2.metric("Accuracy", f"{best['accuracy']:.4f}")
            d3.metric("Output valid", f"{best['output_contract_valid_rate']:.1%}")
            d4.metric("Total tokens", f"{summary['total_tokens'].sum():,.0f}")
            d5.metric("Run wall time", f"{elapsed:.1f}s")
            st.caption(_sample_quality_message(int(best['requests']), float(best['accuracy']), float(best['accuracy_ci_low']), float(best['accuracy_ci_high']), language))
            c1, c2 = st.columns(2)
            c1.plotly_chart(_performance_figure(summary, language), use_container_width=True, key=f"post_f1_{run_id}")
            c2.plotly_chart(_efficiency_figure(summary, language), use_container_width=True, key=f"post_tradeoff_{run_id}")
            diff_fig = _difficulty_figure(summary, language)
            case_fig = _case_heatmap(history_root, summary, language)
            c3, c4 = st.columns(2)
            if diff_fig is not None:
                c3.plotly_chart(diff_fig, use_container_width=True, key=f"post_difficulty_{run_id}")
            if case_fig is not None:
                c4.plotly_chart(case_fig, use_container_width=True, key=f"post_case_{run_id}")
            detail_cols = [
                "strategy_title", "requests", "accuracy", "accuracy_ci_low", "accuracy_ci_high",
                "balanced_accuracy", "macro_f1", "matthews_corrcoef", "cohen_kappa",
                "hard_macro_f1", "output_contract_valid_rate", "mean_total_tokens", "total_tokens",
                "tokens_per_correct_prediction", "p95_latency_seconds", "benchmark_cost_usd",
            ]
            st.dataframe(summary[[c for c in detail_cols if c in summary]], use_container_width=True, hide_index=True)
            if 'paired' in locals() and not paired.empty:
                st.markdown("#### " + ("Páros összehasonlítás P0 baseline-hoz" if language == "hu" else "Paired comparison against P0 baseline"))
                st.caption(
                    "A fixed/regressed számok ugyanazon sample-eken mutatják a valódi javulást és visszaesést; a McNemar exact p csak kiegészítő statisztikai jel, nem önmagában döntési szabály."
                    if language == "hu" else
                    "Fixed/regressed counts compare the exact same samples; McNemar exact p is supporting statistical evidence, not a decision rule by itself."
                )
                st.dataframe(paired, use_container_width=True, hide_index=True)
                st.plotly_chart(
                    px.scatter(paired, x="regressed_vs_p0", y="fixed_vs_p0", size="samples", hover_name="strategy_title",
                               color="paired_accuracy_delta_pp", title="Fixed vs regressed samples"),
                    use_container_width=True, key=f"paired_{run_id}"
                )
        except Exception as exc:
            if 'run_id' in locals():
                mark_failed(run_id, f"{type(exc).__name__}: {exc}")
            st.error(f"{t('failed')}: {type(exc).__name__}: {exc}")

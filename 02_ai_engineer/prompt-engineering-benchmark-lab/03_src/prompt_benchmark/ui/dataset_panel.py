from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from prompt_benchmark.config import load_yaml
from prompt_benchmark.paths import PATHS
from prompt_benchmark.data.benchmark_suites import profile_dataset, prepare_hf_support_router_suite
from prompt_benchmark.data.mock_generator import save_mock_support_tickets
from prompt_benchmark.data.prepare_benchmark import load_sample_dataframe, prepare_splits, save_prepared_data


SCENARIO_HELP_HU = {
    "easy_clear": "egyértelmű intent, erős lexikai jel",
    "implicit_request": "implicit intent, címkeszó nélkül",
    "ambiguous_boundary": "kategóriahatár és erős distractor",
    "multi_intent_primary": "több intent, explicit primary döntés",
    "noisy_typo": "elgépelés, kisbetű, zajos írásmód",
    "long_context": "hosszú irreleváns kontextus",
    "prompt_injection": "ticketbe ágyazott utasítás",
    "resolved_history": "régi lezárt intent + új kérés",
    "negation_correction": "explicit tagadás és korrekció",
    "quoted_thread": "idézett korábbi email thread",
    "multilingual_mixed": "kevert magyar/angol szöveg",
    "telegraphic_short": "nagyon rövid, telegram-stílusú input",
    "primary_last": "több distractor után csak a végén jelenik meg a valódi kérés",
    "primary_first": "valódi kérés az elején, utána erős distractorok",
    "conditional_distractor": "feltételes jövőbeli intent vs. jelenlegi action",
    "code_log_noise": "logok, error code-ok és címkeszavak zajként",
    "label_word_attack": "adversarial szöveg explicit rossz labellel",
    "double_negation": "kettős tagadás és meta-szöveg",
}


def render_dataset_panel(language: str) -> None:
    hu = language == "hu"
    st.subheader("Dataset előkészítés és minőségellenőrzés" if hu else "Dataset preparation and quality check")
    st.write(
        "A Challenge Set most 10× nagyobb: 10 800 offline ticketből készül egy 6 000 soros final holdout és 3 000 soros development set. A 18 scenario family szándékosan tartalmaz ambiguity, multi-intent, injection, negation, log-noise és más nehéz eseteket."
        if hu else
        "The Challenge Set is now 10× larger: 10,800 offline tickets feed a 6,000-row final holdout plus a 3,000-row development set. Eighteen scenario families deliberately cover ambiguity, multi-intent, injection, negation, log noise and other difficult cases."
    )
    st.info(
        "A 6 000 soros holdout a stabil statisztikához kell; cloud API-val először reprezentatív, stratifikált pilot subsetet futtass, ne a teljes 6 000 × stratégia mátrixot."
        if hu else
        "The 6,000-row holdout exists for stable statistics; with cloud APIs start with a representative stratified pilot subset rather than the full 6,000 × strategy matrix."
    )

    benchmark_path = PATHS.processed_data / "benchmark.csv"
    dev_path = PATHS.processed_data / "development.csv"
    raw_path = PATHS.mock_data / "mock_support_tickets.csv"

    if st.button("🔄 10× Challenge Set újragenerálása" if hu else "🔄 Regenerate 10× Challenge Set", key=f"{language}_regen_dataset_v4"):
        cfg = load_yaml("configs/benchmark.yaml")
        source_per_class = int(cfg.get("mock_source_samples_per_class", 1800))
        with st.status("Dataset generálása…" if hu else "Generating dataset…", expanded=True) as status:
            save_mock_support_tickets(raw_path, samples_per_class=source_per_class)
            frame = load_sample_dataframe(raw_path)
            benchmark, development, few_shot = prepare_splits(
                frame,
                int(cfg["benchmark_samples_per_class"]),
                int(cfg["development_samples_per_class"]),
                int(cfg["few_shot_examples_per_class"]),
                int(cfg["random_seed"]),
            )
            save_prepared_data(benchmark, development, few_shot, PATHS.processed_data, PATHS.prompt_examples / "few_shot_examples.json")
            status.write(
                f"raw={len(frame):,} · benchmark={len(benchmark):,} · development={len(development):,} · few-shot={len(few_shot):,}"
            )
            status.update(label="Kész" if hu else "Done", state="complete")
        st.session_state["dataset_ready"] = True
        st.rerun()

    if not benchmark_path.exists():
        st.warning("Nincs előkészített benchmark. Generáld újra fent." if hu else "No prepared benchmark. Regenerate it above.")
        return

    benchmark = pd.read_csv(benchmark_path)
    development = pd.read_csv(dev_path) if dev_path.exists() else pd.DataFrame()
    raw = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    profile = profile_dataset(benchmark, "challenge")
    c = st.columns(7)
    c[0].metric("Raw tickets", f"{len(raw):,}" if len(raw) else "—")
    c[1].metric("Final holdout", f"{profile.rows:,}")
    c[2].metric("Development", f"{len(development):,}")
    c[3].metric("Classes", profile.labels)
    c[4].metric("Avg words", f"{profile.mean_words:.1f}")
    c[5].metric("Hard share", "—" if profile.hard_share is None else f"{profile.hard_share:.1%}")
    c[6].metric("Class balance", f"{profile.min_class_size}/{profile.max_class_size}")

    a, b = st.columns(2)
    counts = benchmark["true_label"].value_counts().rename_axis("label").reset_index(name="count")
    a.plotly_chart(px.bar(counts, x="label", y="count", title="Class distribution"), use_container_width=True)
    if "case_type" in benchmark:
        cases = benchmark["case_type"].value_counts().rename_axis("case_type").reset_index(name="count")
        b.plotly_chart(px.bar(cases, x="case_type", y="count", title="Scenario distribution"), use_container_width=True)

    difficulty = benchmark["difficulty"].value_counts().rename_axis("difficulty").reset_index(name="count") if "difficulty" in benchmark else pd.DataFrame()
    if not difficulty.empty:
        st.plotly_chart(px.bar(difficulty, x="difficulty", y="count", title="Difficulty distribution"), use_container_width=True)

    preview_cols = [c for c in ["sample_id", "text", "true_label", "case_type", "difficulty", "secondary_label"] if c in benchmark]
    st.dataframe(benchmark[preview_cols].head(24), use_container_width=True, hide_index=True)

    st.markdown("### " + ("18 esettípus" if hu else "18 scenario families"))
    if "case_type" in benchmark:
        table = benchmark.groupby(["case_type", "difficulty"]).size().reset_index(name="samples")
        if hu:
            table["miért_fontos"] = table["case_type"].map(SCENARIO_HELP_HU)
        st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown("### " + ("Külső generalizációs suite" if hu else "External generalization suite"))
    hf_path = PATHS.benchmark_suites / "hf_support_router_300.csv"
    if hf_path.exists():
        st.success(("Előkészítve: " if hu else "Prepared: ") + str(hf_path))
    elif st.button("HF generalization suite előkészítése" if hu else "Prepare HF generalization suite", key=f"{language}_dataset_hf_v4"):
        try:
            out, _ = prepare_hf_support_router_suite()
            st.success(str(out))
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")

    st.session_state["dataset_ready"] = True

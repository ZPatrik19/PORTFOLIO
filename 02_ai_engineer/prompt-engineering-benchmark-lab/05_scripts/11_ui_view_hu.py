from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from prompt_benchmark.paths import PATHS
from prompt_benchmark.benchmark.runner import run_strategy
from prompt_benchmark.config import load_yaml
from prompt_benchmark.constants import INVALID_LABEL, LABELS
from prompt_benchmark.evaluation.metrics import classification_metrics, confusion_table, per_class_report
from prompt_benchmark.evaluation.parsing import parse_prediction
from prompt_benchmark.llm.factory import PROVIDER_CAPABILITIES, SUPPORTED_PROVIDERS, create_llm_client
from prompt_benchmark.prompts import (
    TECHNIQUE_CATALOG,
    CustomPromptStrategy,
    get_strategy,
    list_custom_prompts,
    list_strategies,
    load_custom_prompt,
    save_custom_prompt,
)
from prompt_benchmark.utils.pricing import get_provider_pricing
from prompt_benchmark.ui.benchmark_panel import render_benchmark_panel, _performance_figure, _efficiency_figure, _difficulty_figure
from prompt_benchmark.ui.dataset_panel import render_dataset_panel
from prompt_benchmark.ui.playground_panel import render_playground_panel
from prompt_benchmark.ui.workflow_panel import render_workflow_banner, render_history_panel
from prompt_benchmark.ui.run_history import latest_run_id, load_summary as load_history_summary, load_raw as load_history_raw

os.chdir(PROJECT_ROOT)
load_dotenv()

cfg = load_yaml(PATHS.configs / "benchmark.yaml")
pricing = load_yaml(PATHS.configs / "pricing.yaml")
STRATEGIES = list_strategies()
LOGGER = logging.getLogger(__name__)

HU_META: dict[str, dict[str, str]] = {
    "p0_zero_shot": {"title": "Zero-shot baseline", "category": "Alap", "hypothesis": "A minimális instrukció adja a kontroll baseline-t."},
    "p1_definitions": {"title": "Kategóriadefiníciók", "category": "Kontextus", "hypothesis": "A címkék explicit jelentése csökkenti a kategóriák közti bizonytalanságot."},
    "p2_role": {"title": "Role / system prompt", "category": "Persona", "hypothesis": "A rendszer-szintű szerep javíthatja az instrukciókövetést."},
    "p3_few_shot": {"title": "Few-shot példák", "category": "Példák", "hypothesis": "A demonstrációk megtanítják az elvárt input→label leképezést."},
    "p4_constraints": {"title": "Explicit korlátozások", "category": "Instrukció", "hypothesis": "A pozitív és negatív szabályok csökkentik a hibás formátumú kimenetet."},
    "p5_decision_policy": {"title": "Strukturált döntési policy", "category": "Döntési logika", "hypothesis": "A prioritási szabályok segítik az átfedő intentek feloldását."},
    "p6_json": {"title": "Prompt-only JSON", "category": "Formátum", "hypothesis": "A JSON-kérés javítja a gépi feldolgozhatóságot, de nem garantál érvényes JSON-t."},
    "p7_structured_output": {"title": "Schema Structured Output", "category": "Formátum", "hypothesis": "A provider által kikényszerített schema minimalizálja a formázási hibákat."},
    "p8_persona": {"title": "Domain persona", "category": "Persona", "hypothesis": "A specializált persona segíthet domain-specifikus döntésekben."},
    "p9_instruction_context": {"title": "Instruction + Context blokkok", "category": "Prompt anatómia", "hypothesis": "A tagolt prompt csökkenti a task, kontextus és adat összekeverését."},
    "p10_format_audience_tone": {"title": "Format + Audience + Tone", "category": "Prompt anatómia", "hypothesis": "Az explicit célközönség és formátum javíthatja a kimeneti fegyelmet."},
    "p11_delimited_data": {"title": "Elhatárolt / izolált adat", "category": "Robusztusság", "hypothesis": "A delimiterek elkülönítik a ticket adatát az utasításoktól."},
    "p12_contrastive_few_shot": {"title": "Kontrasztív few-shot", "category": "Advanced példák", "hypothesis": "A határeset-példák segítik a könnyen összekeverhető kategóriákat."},
    "p13_reasoning_model": {"title": "Reasoning-model mód", "category": "Reasoning", "hypothesis": "A provider reasoning módja javíthat a nehéz, kétértelmű eseteken."},
    "p14_tree_branch_vote": {"title": "Tree-inspired branch + vote", "category": "Advanced reasoning", "hypothesis": "Független szakértői ágak és többségi szavazás növelheti a robusztusságot extra költségért."},
    "p15_grammar_constrained": {"title": "Grammar/schema constrained", "category": "Constrained generation", "hypothesis": "A constrained decoding javítja a szintaktikai érvényességet."},
    "p16_full_advanced_template": {"title": "Teljes advanced prompt template", "category": "Prompt template", "hypothesis": "A hasznos komponensek kombinálása javíthatja a minőséget, de növeli a tokenigényt."},
}


# -----------------------------------------------------------------------------
# Általános helper-ek
# -----------------------------------------------------------------------------
def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, UnicodeDecodeError, ValueError, pd.errors.ParserError) as exc:
        LOGGER.warning("Could not read CSV %s: %s", path, exc)
        return None


def strategy_title(name: str) -> str:
    meta = HU_META.get(name) or TECHNIQUE_CATALOG.get(name, {})
    prefix = name.split("_")[0].upper() if name.startswith("p") else "CUSTOM"
    return f"{prefix} · {meta.get('title', name)}"


def _active_history_run(provider_name: str) -> str | None:
    selected = st.session_state.get("selected_history_run_id")
    if selected:
        try:
            from prompt_benchmark.ui.run_history import load_manifest
            manifest = load_manifest(str(selected))
            if manifest.get("provider") == provider_name:
                return str(selected)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            LOGGER.warning("Ignoring unavailable selected history run %s: %s", selected, exc)
    return latest_run_id(provider_name)


def load_provider_summary(provider_name: str) -> pd.DataFrame | None:
    run_id = _active_history_run(provider_name)
    frame = load_history_summary(run_id) if run_id else None
    if frame is None:
        frame = safe_read_csv(PATHS.results / provider_name / "benchmark_summary.csv")
    if frame is not None and not frame.empty and "strategy" in frame:
        frame = frame.copy()
        frame["strategy_title"] = frame["strategy"].map(strategy_title)
    return frame


def load_provider_raw(provider_name: str, strategy_name: str) -> pd.DataFrame | None:
    run_id = _active_history_run(provider_name)
    frame = load_history_raw(run_id, strategy_name) if run_id else None
    if frame is not None:
        return frame
    return safe_read_csv(PATHS.results / "raw" / provider_name / f"{strategy_name}.csv")


def run_command(command: list[str], timeout: int = 300) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            env=os.environ.copy(),
        )
        output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        return proc.returncode, output.strip()
    except subprocess.TimeoutExpired:
        return 124, f"Időtúllépés ({timeout} s): {' '.join(command)}"
    except Exception as exc:
        return 1, f"{type(exc).__name__}: {exc}"


def metric_bar(summary: pd.DataFrame, metric: str, title: str, y_title: str) -> go.Figure:
    ordered = summary.sort_values(metric, ascending=False).copy()
    hover = [c for c in ["model", "accuracy", "macro_f1", "p95_latency_seconds", "mean_total_tokens", "cost_per_1000_requests_usd"] if c in ordered]
    fig = px.bar(ordered, x="strategy_title", y=metric, hover_data=hover, title=title)
    fig.update_layout(xaxis_title="Prompt stratégia", yaxis_title=y_title, xaxis_tickangle=-45)
    return fig


def tradeoff(summary: pd.DataFrame, x: str, x_title: str, title: str) -> go.Figure:
    size = summary["mean_total_tokens"].fillna(1).clip(lower=1) if "mean_total_tokens" in summary else None
    fig = px.scatter(
        summary,
        x=x,
        y="macro_f1",
        size=size,
        hover_name="strategy_title",
        hover_data={c: True for c in ["accuracy", "p95_latency_seconds", "mean_total_tokens", "cost_per_1000_requests_usd", "model"] if c in summary},
        title=title,
    )
    fig.update_layout(xaxis_title=x_title, yaxis_title="Macro F1")
    return fig


def confusion_figure(frame: pd.DataFrame, title: str) -> go.Figure:
    """NaN/invalid outputot külön oszlopban jelenít meg, ezért nem crash-el."""
    matrix = confusion_table(frame)
    return px.imshow(
        matrix,
        x=list(matrix.columns),
        y=list(matrix.index),
        text_auto=True,
        aspect="auto",
        labels={"x": "Predikált címke", "y": "Valódi címke", "color": "Darab"},
        title=title,
    )


def hu_template_path(strategy_name: str) -> Path:
    return PATHS.prompts / "hu" / f"{strategy_name}.txt"


def render_hu_template(strategy_name: str, ticket: str) -> str:
    path = hu_template_path(strategy_name)
    if not path.exists():
        return "Magyar tükörtemplate még nem található."
    text = path.read_text(encoding="utf-8")
    return text.replace("{ticket}", ticket)


def custom_options() -> dict[str, Path]:
    return {p.stem: p for p in list_custom_prompts()}


def build_custom_from_state(prefix: str = "custom") -> CustomPromptStrategy:
    return CustomPromptStrategy(
        display_name=st.session_state.get(f"{prefix}_name", "sajat_prompt"),
        system_prompt=st.session_state.get(f"{prefix}_system", ""),
        user_template=st.session_state.get(f"{prefix}_user", "Ticket:\n{ticket}\n\nCsak a címkét add vissza."),
        output_mode=st.session_state.get(f"{prefix}_output_mode", "label"),
        structured_output=bool(st.session_state.get(f"{prefix}_structured", False)),
        reasoning_effort=(st.session_state.get(f"{prefix}_reasoning") or None),
    )


def provider_preflight(provider_name: str, model_name: str) -> tuple[bool, str]:
    if provider_name == "mock":
        return True, "A Mock provider nem igényel külső szolgáltatást."
    if provider_name == "ollama":
        return True, "Az Ollama nem igényel API key-t, de az Ollama szolgáltatásnak futnia kell."
    env_by_provider = {
        "groq": "GROQ_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    env_name = env_by_provider.get(provider_name)
    if env_name and not os.getenv(env_name):
        return False, f"Hiányzik a {env_name}. Add meg a bal oldali sávban vagy a .env fájlban."
    if not model_name or model_name.startswith("<"):
        return False, "Nincs érvényes modellazonosító megadva."
    return True, "Konfiguráció rendben; a tényleges elérhetőséget egy próbahívás ellenőrzi."


# -----------------------------------------------------------------------------
# Fejléc — a runtime/provider beállításokat a közös 11_ui_app.py kezeli.
# -----------------------------------------------------------------------------
st.title("🧪 Prompt Engineering Benchmark Lab")
st.caption("Magyar interaktív prompt engineering labor: benchmark, custom prompt, token/latency/cost mérés, output validáció és API-integráció.")

def client_overrides() -> dict[str, object]:
    values: dict[str, object] = {"model": model}
    if use_temperature:
        values["temperature"] = temperature
    if use_top_p:
        values["top_p"] = top_p
    if use_top_k:
        values["top_k"] = top_k
    return values


def build_client(extra_overrides: dict[str, object] | None = None):
    ok, msg = provider_preflight(provider, model)
    if not ok:
        raise RuntimeError(msg)
    overrides = client_overrides()
    overrides.update(extra_overrides or {})
    return create_llm_client(provider, cfg, overrides=overrides)


provider_ok, _provider_msg = provider_preflight(provider, model)
render_workflow_banner("hu", provider_ok)

(
    tab_dataset,
    tab_playground,
    tab_benchmark,
    tab_validation,
    tab_dashboard,
    tab_history,
    tab_techniques,
    tab_custom,
    tab_sweep,
    tab_finetune,
    tab_api,
    tab_system,
) = st.tabs([
    "2️⃣ Dataset",
    "3️⃣ Playground",
    "4️⃣ Benchmark",
    "5️⃣ Output validáció",
    "6️⃣ Dashboard",
    "7️⃣ History",
    "🧠 Prompttechnikák",
    "✍️ Saját prompt",
    "🎛 Paraméterlabor",
    "🧬 Fine-tuning",
    "🔌 API-integráció",
    "🛠 Rendszer",
])

with tab_dataset:
    render_dataset_panel("hu")


# -----------------------------------------------------------------------------
# Dashboard
# -----------------------------------------------------------------------------
with tab_dashboard:
    st.subheader(f"Mentett benchmark eredmények — {provider}")
    active_run = _active_history_run(provider)
    h1, h2 = st.columns([5, 1])
    h1.caption(f"Aktív history run: `{active_run or 'legacy/latest'}`")
    if h2.button("🔄 Frissítés", key="hu_dashboard_refresh_v3"):
        st.rerun()
    summary = load_provider_summary(provider)
    if summary is None or summary.empty:
        st.info("Ehhez a providerhez még nincs mentett benchmark. Futtass benchmarkot, majd generálj riportot.")
    else:
        best = summary.sort_values("macro_f1", ascending=False).iloc[0]
        baseline_rows = summary[summary["strategy"] == "p0_zero_shot"]
        baseline = baseline_rows.iloc[0] if not baseline_rows.empty else best
        fastest = summary.sort_values("p95_latency_seconds").iloc[0]
        reliable = summary.sort_values(["invalid_output_rate", "macro_f1"], ascending=[True, False]).iloc[0]

        cols = st.columns(6)
        cols[0].metric("Legjobb Macro F1", f"{best['macro_f1']:.4f}", strategy_title(str(best['strategy'])))
        cols[1].metric("P0 baseline", f"{baseline['macro_f1']:.4f}")
        cols[2].metric("Javulás", f"{(best['macro_f1']-baseline['macro_f1'])*100:+.2f} pp")
        cols[3].metric("Output validitás", f"{1-reliable['invalid_output_rate']:.1%}")
        cols[4].metric("Leggyorsabb P95", f"{fastest['p95_latency_seconds']:.3f}s")
        cols[5].metric("Best token/request", f"{best['mean_total_tokens']:.0f}")

        sample_n = int(best.get("requests", 0))
        if {"accuracy_ci_low", "accuracy_ci_high"}.issubset(summary.columns):
            st.info(
                f"Aktív run: n={sample_n} / stratégia · best Accuracy={best['accuracy']:.3f} · "
                f"Wilson 95% CI=[{best['accuracy_ci_low']:.3f}, {best['accuracy_ci_high']:.3f}]. "
                + ("Ez még pilot minta; az 1.000 score nem tekinthető stabil final eredménynek." if sample_n < 60 else "A mintaszám már alkalmasabb összehasonlításra, de a scenario bontást is nézd." )
            )
        a, b = st.columns(2)
        a.plotly_chart(_performance_figure(summary, "hu"), use_container_width=True)
        b.plotly_chart(_efficiency_figure(summary, "hu"), use_container_width=True)
        diff_fig = _difficulty_figure(summary, "hu")
        a, b = st.columns(2)
        if diff_fig is not None:
            a.plotly_chart(diff_fig, use_container_width=True)
        b.plotly_chart(tradeoff(summary, "p95_latency_seconds", "P95 latency (s)", "Minőség vs latency"), use_container_width=True)

        st.markdown("### Lényeges számok és futási beállítások")
        show = [
            "strategy_title", "model", "requests", "accuracy", "accuracy_ci_low", "accuracy_ci_high", "balanced_accuracy", "matthews_corrcoef", "cohen_kappa", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "hard_macro_f1",
            "output_contract_valid_rate", "invalid_json_rate", "mean_input_tokens", "mean_output_tokens", "mean_total_tokens",
            "total_tokens", "tokens_per_correct_prediction", "p50_latency_seconds", "p95_latency_seconds", "p99_latency_seconds",
            "cost_per_1000_requests_usd", "benchmark_cost_usd", "temperature", "top_p", "top_k", "mean_branch_count",
            "token_source", "latency_source",
        ]
        st.dataframe(summary[[c for c in show if c in summary]], use_container_width=True, hide_index=True)

        case_path = (PATHS.results / "history" / active_run / "case_type_summary.csv") if active_run else (PATHS.results / provider / "case_type_summary.csv")
        case_frame = safe_read_csv(case_path)
        if case_frame is not None and not case_frame.empty:
            case_frame["strategy_title"] = case_frame["strategy"].map(strategy_title)
            pivot = case_frame.pivot(index="strategy_title", columns="case_type", values="accuracy")
            st.plotly_chart(px.imshow(pivot, text_auto=".3f", zmin=0, zmax=1, aspect="auto", title="Pontosság esettípusonként"), use_container_width=True)

        paired_path = (PATHS.results / "history" / active_run / "paired_vs_p0.csv") if active_run else None
        paired = safe_read_csv(paired_path) if paired_path else None
        if paired is not None and not paired.empty:
            if "strategy" in paired:
                paired["strategy_title"] = paired["strategy"].map(strategy_title)
            st.markdown("### Páros P0 baseline összehasonlítás")
            st.caption("Ugyanazon sample-eken: hány hibát javított ki az új prompt, és hány korábban jó esetet rontott el.")
            st.dataframe(paired, use_container_width=True, hide_index=True)
            st.plotly_chart(px.scatter(paired, x="regressed_vs_p0", y="fixed_vs_p0", color="paired_accuracy_delta_pp", hover_name="strategy_title", title="Javított vs regresszált sample-ek"), use_container_width=True)


# -----------------------------------------------------------------------------
# Technikák + teljes template galéria
# -----------------------------------------------------------------------------
with tab_techniques:
    st.subheader("Prompttechnika könyvtár és teljes prompt-template galéria")
    catalog = pd.DataFrame([
        {"stratégia": name, "név": strategy_title(name), "kategória": HU_META[name]["category"], "hipotézis": HU_META[name]["hypothesis"]}
        for name in STRATEGIES
    ])
    st.dataframe(catalog, use_container_width=True, hide_index=True)

    sample = st.text_area("Minta ticket a promptok rendereléséhez", "I was charged twice, but my main request is to cancel before the next renewal.", key="hu_preview_ticket")
    strategy_name = st.selectbox("Prompttechnika", STRATEGIES, format_func=strategy_title, key="hu_template_strategy")
    base = get_strategy(strategy_name)
    payloads = base.build_branches(sample) if hasattr(base, "build_branches") else [base.build(sample)]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### A benchmark által ténylegesen használt prompt")
        for i, payload in enumerate(payloads, 1):
            if len(payloads) > 1:
                st.markdown(f"**Ág {i}**")
            st.code(f"SYSTEM:\n{payload.instructions or '(nincs)'}\n\nUSER:\n{payload.input_text}", language="text")
        chars = sum(len((p.instructions or "") + p.input_text) for p in payloads)
        words = sum(len(((p.instructions or "") + " " + p.input_text).split()) for p in payloads)
        st.caption(f"Ágak: {len(payloads)} · karakter: {chars:,} · szó: {words:,} · durva tokenbecslés: {chars/4:.0f}")
    with c2:
        st.markdown("#### Magyar tükörtemplate")
        hu_text = render_hu_template(strategy_name, sample)
        st.code(hu_text, language="text")
        if st.button("Magyar template betöltése a Saját prompt szerkesztőbe", key="load_hu_custom"):
            st.session_state["custom_name"] = f"hu_{strategy_name}"
            st.session_state["custom_system"] = "Te egy precíz SaaS ügyfélszolgálati routing osztályozó vagy."
            raw_template = hu_template_path(strategy_name).read_text(encoding="utf-8") if hu_template_path(strategy_name).exists() else "Ticket: {ticket}"
            st.session_state["custom_user"] = raw_template
            sample_payload = payloads[0]
            st.session_state["custom_output_mode"] = sample_payload.output_mode
            st.session_state["custom_structured"] = sample_payload.structured_output
            st.session_state["custom_reasoning"] = sample_payload.reasoning_effort or ""
            st.success("Betöltve. Nyisd meg a 'Saját prompt' fület.")

    with st.expander("Összes P0–P16 magyar template"):
        for name in STRATEGIES:
            st.markdown(f"**{strategy_title(name)}**")
            st.code(render_hu_template(name, "{ticket}"), language="text")


# -----------------------------------------------------------------------------
# Custom prompt szerkesztő
# -----------------------------------------------------------------------------
with tab_custom:
    st.subheader("Saját prompt szerkesztő, mentés és benchmark")
    st.write("A `{ticket}` helyőrző kötelező. JSON kapcsos zárójelekhez használj `{{` és `}}` escape-et a Python template miatt.")

    st.session_state.setdefault("custom_name", "sajat_advanced_prompt")
    st.session_state.setdefault("custom_system", "Te egy precíz SaaS support-routing osztályozó vagy.")
    st.session_state.setdefault("custom_user", "Feladat: osztályozd az alábbi ticket elsődleges szándékát.\nEngedélyezett címkék: api, billing, cancellation, complaint, technical, upgrade.\n\n<ticket>\n{ticket}\n</ticket>\n\nCsak a végső címkét add vissza.")
    st.session_state.setdefault("custom_output_mode", "label")
    st.session_state.setdefault("custom_structured", False)
    st.session_state.setdefault("custom_reasoning", "")

    saved = custom_options()
    chosen_saved = st.selectbox("Mentett preset betöltése", ["(új prompt)", *saved.keys()], key="custom_saved_choice")
    if chosen_saved != "(új prompt)" and st.button("Preset betöltése", key="custom_load_button"):
        loaded = load_custom_prompt(saved[chosen_saved])
        st.session_state["custom_name"] = loaded.display_name
        st.session_state["custom_system"] = loaded.system_prompt
        st.session_state["custom_user"] = loaded.user_template
        st.session_state["custom_output_mode"] = loaded.output_mode
        st.session_state["custom_structured"] = loaded.structured_output
        st.session_state["custom_reasoning"] = loaded.reasoning_effort or ""
        st.rerun()

    st.text_input("Prompt neve", key="custom_name")
    st.text_area("System prompt", height=120, key="custom_system")
    st.text_area("User prompt template", height=260, key="custom_user")
    a, b, c = st.columns(3)
    a.selectbox("Output mód", ["label", "json"], key="custom_output_mode")
    b.checkbox("Structured Output / schema", key="custom_structured")
    c.selectbox("Reasoning effort", ["", "low", "medium", "high"], key="custom_reasoning")

    try:
        custom_strategy = build_custom_from_state()
        preview_ticket = st.text_area("Preview ticket", "Please cancel my subscription before the next renewal.", key="custom_preview_ticket")
        preview_payload = custom_strategy.build(preview_ticket)
        st.markdown("#### Renderelt prompt")
        st.code(f"SYSTEM:\n{preview_payload.instructions or '(nincs)'}\n\nUSER:\n{preview_payload.input_text}", language="text")
        prompt_chars = len((preview_payload.instructions or "") + preview_payload.input_text)
        c1, c2, c3 = st.columns(3)
        c1.metric("Karakter", f"{prompt_chars:,}")
        c2.metric("Durva tokenbecslés", f"{prompt_chars/4:.0f}")
        c3.metric("Structured", "Igen" if preview_payload.structured_output else "Nem")

        if st.button("💾 Custom prompt mentése", key="save_custom"):
            path = save_custom_prompt(custom_strategy)
            st.success(f"Mentve: {path}")

        left, right = st.columns(2)
        with left:
            if st.button("▶ Egy ticket futtatása", type="primary", key="run_custom_single"):
                try:
                    client = build_client()
                    response = client.classify(preview_payload)
                    parsed = parse_prediction(response.raw_output, preview_payload.output_mode)
                    st.metric("Predikció", parsed.label or INVALID_LABEL)
                    st.code(response.raw_output or "(üres válasz)", language="json" if preview_payload.output_mode == "json" else "text")
                    st.json({
                        "valid_output": parsed.valid_output,
                        "valid_json": parsed.valid_json,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "total_tokens": response.total_tokens,
                        "latency_seconds": response.latency_seconds,
                        "provider": response.provider,
                        "model": response.model,
                        "error": response.error,
                    })
                except Exception as exc:
                    st.error(f"Futtatási hiba: {type(exc).__name__}: {exc}")
        with right:
            custom_limit = st.number_input("Custom benchmark mintaszám", 6, 300, 48, 6, key="custom_bench_limit")
            if st.button("🏁 Custom prompt mini benchmark", key="run_custom_bench"):
                try:
                    data = pd.read_csv(PATHS.processed_data / "benchmark.csv")
                    client = build_client()
                    inp, outp = get_provider_pricing(pricing, provider)
                    output_path = PATHS.results / "ui_runs" / provider / f"{custom_strategy.name}.csv"
                    result = run_strategy(data, custom_strategy, client, output_path, inp, outp, limit=int(custom_limit), force=True)
                    metrics = classification_metrics(result)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                    c2.metric("Macro F1", f"{metrics['macro_f1']:.4f}")
                    c3.metric("Token/request", f"{metrics['mean_total_tokens']:.0f}")
                    c4.metric("P95 latency", f"{metrics['p95_latency_seconds']:.3f}s")
                    st.plotly_chart(confusion_figure(result, "Custom prompt confusion matrix"), use_container_width=True)
                except Exception as exc:
                    st.error(f"Benchmark hiba: {type(exc).__name__}: {exc}")
    except Exception as exc:
        st.warning(str(exc))


# -----------------------------------------------------------------------------
# Playground
# -----------------------------------------------------------------------------
with tab_playground:
    render_playground_panel(
        language="hu",
        provider=provider,
        model=model,
        strategies=STRATEGIES,
        strategy_title=strategy_title,
        build_client=build_client,
    )


with tab_benchmark:
    render_benchmark_panel(
        language="hu",
        provider=provider,
        model=model,
        strategies=STRATEGIES,
        strategy_title=strategy_title,
        build_client=build_client,
        pricing_config=pricing,
    )


# -----------------------------------------------------------------------------
# Paraméterlabor
# -----------------------------------------------------------------------------
with tab_sweep:
    st.subheader("Decoding paraméterlabor")
    st.write("A prompt fix marad, és külön változtatjuk a `temperature`, `top_p`, `top_k` értékeket, hogy ne keverjük össze a prompt-design és sampling hatását.")
    sweep_strategy = st.selectbox("Fix promptstratégia", STRATEGIES, index=STRATEGIES.index("p16_full_advanced_template"), format_func=strategy_title, key="hu_sweep_strategy")
    sweep_limit = st.slider("Minták / beállítás", 6, 120, 24, 6)
    supported = [name for name in ["temperature", "top_p", "top_k"] if caps.get(name)]
    st.info("Támogatott paraméterek ennél a providernél: " + (", ".join(supported) if supported else "nincs"))
    if st.button("Parameter sweep futtatása", disabled=not supported):
        code, output = run_command([
            sys.executable, "05_scripts/06_run_parameter_sweep.py", "--provider", provider,
            "--strategy", sweep_strategy, "--limit", str(sweep_limit), "--force",
        ])
        st.code(output or "(nincs konzolkimenet)")
        st.success("Parameter sweep kész.") if code == 0 else st.error("Parameter sweep hiba.")
    sweep = safe_read_csv(PATHS.results / "parameter_sweeps" / provider / "parameter_sweep_summary.csv")
    if sweep is not None and not sweep.empty:
        metric = st.selectbox("Megjelenített metrika", ["macro_f1", "accuracy", "invalid_output_rate", "mean_total_tokens", "p95_latency_seconds"])
        fig = px.line(sweep.sort_values(["parameter", "value"]), x="value", y=metric, color="parameter", markers=True, hover_data=[c for c in ["model", "strategy", "mean_total_tokens", "p95_latency_seconds"] if c in sweep], title=f"Decoding érzékenység — {metric}")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(sweep, use_container_width=True, hide_index=True)
    else:
        st.info("Még nincs mentett parameter sweep ehhez a providerhez.")


# -----------------------------------------------------------------------------
# Output Validation — crash-safe
# -----------------------------------------------------------------------------
with tab_validation:
    st.subheader("Output validáció és hibaanalízis")
    available_raw = [s for s in STRATEGIES if load_provider_raw(provider, s) is not None]
    if not available_raw:
        st.info("Előbb futtass legalább egy canonical benchmark stratégiát.")
    else:
        selected = st.selectbox("Vizsgált stratégia", available_raw, format_func=strategy_title, key="hu_validation_strategy")
        raw = load_provider_raw(provider, selected)
        if raw is None or raw.empty:
            st.warning("A raw result fájl üres vagy nem olvasható.")
        else:
            try:
                metrics = classification_metrics(raw)
                c = st.columns(6)
                c[0].metric("Accuracy", f"{metrics['accuracy']:.4f}")
                c[1].metric("Macro F1", f"{metrics['macro_f1']:.4f}")
                c[2].metric("Output valid", f"{metrics['output_contract_valid_rate']:.1%}")
                c[3].metric("Invalid output", f"{metrics['invalid_output_rate']:.1%}")
                c[4].metric("Invalid JSON", "N/A" if pd.isna(metrics['invalid_json_rate']) else f"{metrics['invalid_json_rate']:.1%}")
                c[5].metric("P95 latency", f"{metrics['p95_latency_seconds']:.3f}s")
                a, b = st.columns(2)
                a.plotly_chart(confusion_figure(raw, f"Confusion matrix — {strategy_title(selected)}"), use_container_width=True)
                report = per_class_report(raw).reset_index().rename(columns={"index": "label", "f1-score": "f1"})
                b.plotly_chart(px.bar(report, x="label", y=["precision", "recall", "f1"], barmode="group", title="Precision / Recall / F1 kategóriánként"), use_container_width=True)

                pred = raw["predicted_label"].fillna(INVALID_LABEL).astype(str)
                valid = raw["valid_output"].fillna(False).astype(bool)
                errors = raw[(~valid) | (pred != raw["true_label"].astype(str))].copy()
                st.markdown(f"#### Hibás / invalid esetek: {len(errors)} / {len(raw)}")
                cols = [c for c in ["sample_id", "case_type", "difficulty", "text", "true_label", "predicted_label", "raw_response", "valid_output", "valid_json", "error"] if c in errors]
                st.dataframe(errors[cols], use_container_width=True, hide_index=True)

                if "latency_seconds" in raw:
                    st.plotly_chart(px.histogram(raw, x="latency_seconds", nbins=30, title="Latency eloszlás"), use_container_width=True)
                if "total_tokens" in raw:
                    st.plotly_chart(px.histogram(raw, x="total_tokens", nbins=30, title="Tokenfogyasztás eloszlása"), use_container_width=True)
            except Exception as exc:
                st.error(f"Output Validation hiba: {type(exc).__name__}: {exc}")
                st.caption("A tab most már a hibát lokalizáltan kezeli, nem állítja le az egész UI-t.")


# -----------------------------------------------------------------------------
# Fine-tuning prep — robust
# -----------------------------------------------------------------------------
with tab_finetune:
    st.subheader("Fine-tuning előkészítés")
    st.write("A fine-tuning train/validation kizárólag a development splitből készül. A final benchmark holdout nem kerül training adatba.")
    dev = safe_read_csv(PATHS.processed_data / "development.csv")
    train_path = PATHS.data / "fine_tuning" / "sft_train.jsonl"
    val_path = PATHS.data / "fine_tuning" / "sft_validation.jsonl"
    c = st.columns(4)
    c[0].metric("Development minta", 0 if dev is None else len(dev))
    c[1].metric("Train JSONL", "Megvan" if train_path.exists() else "Hiányzik")
    c[2].metric("Validation JSONL", "Megvan" if val_path.exists() else "Hiányzik")
    c[3].metric("Holdout leakage", "NINCS")

    if st.button("Leakage-safe SFT JSONL újragenerálása", key="hu_ft_export"):
        code, output = run_command([sys.executable, "05_scripts/12_export_finetuning_data.py"])
        st.code(output or "(nincs konzolkimenet)")
        st.success("Fine-tuning dataset elkészült.") if code == 0 else st.error("Fine-tuning export sikertelen.")

    def jsonl_preview(path: Path, n: int = 3) -> list[dict]:
        if not path.exists():
            return []
        rows = []
        try:
            with path.open(encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if i >= n:
                        break
                    rows.append(json.loads(line))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            LOGGER.warning("Could not preview fine-tuning JSONL %s: %s", path, exc)
            return []
        return rows

    with st.expander("SFT train preview"):
        preview = jsonl_preview(train_path)
        st.json(preview if preview else {"status": "Nincs olvasható train JSONL. Futtasd az exportot."})
    st.code("python 05_scripts/04_run_benchmark.py --provider <provider> --model <fine-tuned-model-id> --strategy p16_full_advanced_template", language="bash")
    st.info("A projekt szándékosan nem indít automatikus fizetős fine-tuning jobot. Az exportált JSONL viszont készen áll provider-specifikus SFT/LoRA workflowhoz.")


# -----------------------------------------------------------------------------
# API integráció + ingyenes szolgáltatók
# -----------------------------------------------------------------------------
with tab_api:
    st.subheader("API-integráció és ingyenes LLM lehetőségek")
    free_table = pd.DataFrame([
        {"Provider": "Ollama", "Ár": "0 USD API-díj", "API key": "Nem", "Ajánlott modell": "llama3.2:3b / saját", "Megjegyzés": "Lokális; saját hardvert használ."},
        {"Provider": "Groq Free Plan", "Ár": "0 USD free quota", "API key": "Igen", "Ajánlott modell": "openai/gpt-oss-20b", "Megjegyzés": "Gyors cloud inference; rate limitet figyelni kell."},
        {"Provider": "Gemini Free Tier", "Ár": "0 USD free tier", "API key": "Igen", "Ajánlott modell": "gemini-3.5-flash-lite", "Megjegyzés": "Google pricing szerint Standard Free Tierben input/output díjmentes a limiteken belül."},
        {"Provider": "OpenRouter Free", "Ár": "$0/M token a free routernél", "API key": "Igen", "Ajánlott modell": "openrouter/free", "Megjegyzés": "A háttérmodell változhat; explorációhoz kiváló."},
        {"Provider": "Hugging Face", "Ár": "kis havi free credit", "API key": "Igen", "Ajánlott modell": "providerfüggő", "Megjegyzés": "Ajánlásként szerepel; külön adapter nincs ebben a verzióban."},
    ])
    st.dataframe(free_table, use_container_width=True, hide_index=True)
    st.caption("Free tier limitek változhatnak. Részletes, dátumozott forráslista: docs/INGYENES_LLM_APIK_HU.md")

    st.markdown("#### Aktuális runtime")
    st.json({
        "provider": provider,
        "model": model,
        "temperature": temperature if use_temperature else None,
        "top_p": top_p if use_top_p else None,
        "top_k": top_k if use_top_k else None,
        "capabilities": caps,
    })

    if st.button("Provider kapcsolat tesztelése", key="hu_api_test"):
        try:
            client = build_client()
            payload = get_strategy("p0_zero_shot").build("I was charged twice for my subscription.")
            response = client.classify(payload)
            parsed = parse_prediction(response.raw_output, payload.output_mode)
            if response.error:
                st.error(response.error)
            else:
                st.success("Provider válaszolt.")
            st.json({
                "raw": response.raw_output,
                "parsed": parsed.label,
                "valid": parsed.valid_output,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "latency_seconds": response.latency_seconds,
                "provider": response.provider,
                "model": response.model,
                "error": response.error,
            })
        except Exception as exc:
            st.error(f"Kapcsolati hiba: {type(exc).__name__}: {exc}")

    if provider == "gemini":
        st.markdown("#### Gemini-specifikus ellenőrzés")
        st.caption("A teszt a Google REST végpontját közvetlenül hívja, a kulcsot soha nem írja ki. A kulcsot a közös sidebar GEMINI_API_KEY mezőjében add meg.")
        if st.button("Gemini kulcs + modell teszt", key="hu_gemini_rest_test"):
            code, output = run_command([sys.executable, "05_scripts/15_test_gemini_connection.py", "--model", model], timeout=60)
            st.code(output or "(nincs kimenet)")
            st.success("Gemini REST kapcsolat OK.") if code == 0 else st.error("Gemini teszt sikertelen. Ellenőrizd a kulcsot, internetet, kvótát és a modellazonosítót.")

    st.markdown("#### Élő API request")
    live_source = st.radio("Prompt", ["Beépített", "Mentett custom"], horizontal=True, key="api_prompt_source")
    live_strategy = None
    if live_source == "Beépített":
        live_name = st.selectbox("Stratégia", STRATEGIES, index=STRATEGIES.index("p16_full_advanced_template"), format_func=strategy_title, key="hu_api_strategy")
        live_strategy = get_strategy(live_name)
    else:
        presets = custom_options()
        if presets:
            preset_name = st.selectbox("Custom preset", list(presets), key="hu_api_custom")
            live_strategy = load_custom_prompt(presets[preset_name])
        else:
            st.warning("Nincs mentett custom prompt.")
    live_ticket = st.text_area("Ticket", "I was charged twice last week. The billing issue is resolved; now I want to cancel before renewal.", key="hu_api_ticket")
    expected = st.selectbox("Várt címke (opcionális)", ["(ismeretlen)", *LABELS], key="hu_api_expected")
    if st.button("Élő request küldése", type="primary", disabled=live_strategy is None, key="hu_api_send"):
        try:
            client = build_client()
            payload = live_strategy.build(live_ticket)
            response = client.classify(payload)
            parsed = parse_prediction(response.raw_output, payload.output_mode)
            input_price, output_price = get_provider_pricing(pricing, provider)
            estimated_cost = response.input_tokens / 1_000_000 * input_price + response.output_tokens / 1_000_000 * output_price
            c = st.columns(6)
            c[0].metric("Predikció", parsed.label or INVALID_LABEL)
            c[1].metric("Input token", response.input_tokens)
            c[2].metric("Output token", response.output_tokens)
            c[3].metric("Összes token", response.total_tokens)
            c[4].metric("Latency", f"{response.latency_seconds:.3f}s")
            c[5].metric("Becsült költség", f"${estimated_cost:.6f}")
            st.code(response.raw_output or "(üres válasz)", language="json" if payload.output_mode == "json" else "text")
            st.json({
                "valid_output": parsed.valid_output,
                "valid_json": parsed.valid_json,
                "structured_output": payload.structured_output,
                "reasoning_effort": payload.reasoning_effort,
                "provider": response.provider,
                "model": response.model,
                "token_source": response.token_source,
                "latency_source": response.latency_source,
                "error": response.error,
            })
            if response.error:
                st.error(response.error)
            if expected != "(ismeretlen)":
                st.success("Egyezik a várt címkével.") if parsed.label == expected else st.error(f"Várt: {expected}; kapott: {parsed.label}")
        except Exception as exc:
            st.error(f"API Integration hiba: {type(exc).__name__}: {exc}")


# -----------------------------------------------------------------------------
# Rendszer / diagnosztika
# -----------------------------------------------------------------------------
with tab_history:
    render_history_panel("hu", provider, strategy_title)


with tab_system:
    st.subheader("Rendszerállapot és diagnosztika")
    checks: Iterable[tuple[str, bool, str]] = [
        ("Virtuális környezet", (PROJECT_ROOT / ".venv").exists(), ".venv"),
        ("Mock raw data", (PATHS.mock_data / "mock_support_tickets.csv").exists(), "01_data/mock/mock_support_tickets.csv"),
        ("Development split", (PATHS.processed_data / "development.csv").exists(), "01_data/processed/development.csv"),
        ("Benchmark split", (PATHS.processed_data / "benchmark.csv").exists(), "01_data/processed/benchmark.csv"),
        ("Magyar prompt template-ek", Path("configs/prompts/hu/p16_full_advanced_template.txt").exists(), "configs/prompts/hu"),
        ("Custom prompt mappa", PATHS.custom_prompts.exists(), "configs/prompts/custom"),
        ("Eredmények", PATHS.results.exists(), "07_outputs/results"),
        ("Riportok", PATHS.reports.exists(), "07_outputs/reports"),
    ]
    status = pd.DataFrame([{"komponens": n, "állapot": "OK" if ok else "HIÁNYZIK", "útvonal": p} for n, ok, p in checks])
    st.dataframe(status, use_container_width=True, hide_index=True)

    st.markdown("#### Környezet")
    st.code(f"Python: {sys.version}\nExecutable: {sys.executable}\nProject root: {PROJECT_ROOT}")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Dependency ellenőrzés", key="hu_dep_check"):
            code, output = run_command([sys.executable, "00_setup/00_dependency_manager.py", "--check-only"], timeout=180)
            st.code(output or "(nincs kimenet)")
            st.success("Dependency check OK.") if code == 0 else st.warning("Van hiányzó vagy frissítendő dependency. Windows alatt futtasd: 00_setup\\05_update_environment.bat")
    with c2:
        if st.button("Gyors pytest", key="hu_pytest"):
            code, output = run_command([sys.executable, "-m", "pytest", "04_tests", "-q"], timeout=180)
            st.code(output or "(nincs kimenet)")
            st.success("Tesztek OK.") if code == 0 else st.error("Van hibás teszt.")
    with c3:
        if st.button("Provider smoke test", key="hu_provider_smoke"):
            try:
                client = build_client()
                payload = get_strategy("p0_zero_shot").build("The application crashes on startup.")
                response = client.classify(payload)
                st.success("Provider válaszolt.") if not response.error else st.error(response.error)
                st.code(response.raw_output or "(üres)")
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")

    st.markdown("#### Windows indítók")
    st.code("RUN_UI.bat\n00_setup\\01_setup_windows.bat\n00_setup\\02_run_ui.bat\n00_setup\\05_update_environment.bat", language="text")

from __future__ import annotations

import json
import os
import sys
import tempfile
import subprocess
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
UI_DIR = Path(__file__).resolve().parent
SRC = ROOT / "03_src"
for local_path in (SRC, UI_DIR):
    if str(local_path) not in sys.path:
        sys.path.insert(0, str(local_path))

from travel_agent.agent import MLRouterTravelAgent, OpenAITravelAgent, PlanThenExecuteTravelAgent
from travel_agent.evaluation.runner import evaluate_agent
from travel_agent.quality import audit_all
from travel_agent.presets import PRESET_QUESTIONS, CATEGORY_LABELS, preset_text
from travel_agent.tools import ToolRegistry
from travel_agent.training import model_status, train_router
from travel_agent.usage import UsageStore
from project_statistics_dashboard import render_dashboard
from data_quality_dashboard import render_data_quality_dashboard
from live_statistics_dashboard import render_live_statistics_dashboard

st.set_page_config(page_title="Agentic Travel Research Assistant", page_icon="🧭", layout="wide")


@st.cache_resource
def get_usage_store() -> UsageStore:
    return UsageStore()


usage_store = get_usage_store()


def get_agent(methodology: str, language: str):
    if methodology == "plan_execute":
        return PlanThenExecuteTravelAgent(language=language)
    if methodology == "ml_router":
        return MLRouterTravelAgent(language=language)
    if methodology == "openai_direct":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not configured in the environment/.env file.")
        return OpenAITravelAgent(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
            max_steps=int(os.getenv("MAX_AGENT_STEPS", "8")),
            language=language,
        )
    raise ValueError(methodology)


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "01_data" / "raw" / name)


@st.cache_data(show_spinner=False)
def load_quality_report() -> dict:
    path = ROOT / "06_results" / "data_quality" / "data_quality_report.json"
    if not path.exists():
        return audit_all(write_outputs=True)
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_project_statistics() -> dict:
    path = ROOT / "06_results" / "project_statistics" / "project_statistics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def show_trace(run) -> None:
    if not run.trace:
        st.info("No tool calls were executed.")
        return
    rows = []
    for call in run.trace:
        rows.append({
            "step": call.step, "tool": call.name, "success": call.success,
            "latency_ms": round(call.latency_ms, 2), "arguments": json.dumps(call.arguments, ensure_ascii=False),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    for call in run.trace:
        with st.expander(f"Step {call.step} · {call.name} · {'OK' if call.success else 'ERROR'}"):
            left, right = st.columns(2)
            with left:
                st.markdown("**Arguments**")
                st.json(call.arguments)
            with right:
                st.markdown("**Output**")
                st.json(call.output)
            st.caption(f"Latency: {call.latency_ms:.2f} ms")


st.title("🧭 Agentic Travel Research Assistant")
st.caption("Local tool-calling playground · data-quality audit · model training · evaluation")

with st.sidebar:
    st.header("Runtime")
    language = st.selectbox("Language", ["hu", "en"], index=0)
    methodology = st.selectbox("Methodology", ["plan_execute", "ml_router", "openai_direct"], index=0)
    data_mode = st.selectbox("Travel data mode", ["local", "auto", "live"], index=0)
    chart_theme = st.selectbox(
        "Analitikai diagramtéma" if language == "hu" else "Analytics chart theme",
        ["light", "dark"],
        format_func=lambda value: ({
            "light": "☀️ Világos · magas kontraszt",
            "dark": "🌙 Sötét · magas kontraszt",
        } if language == "hu" else {
            "light": "☀️ Light · high contrast",
            "dark": "🌙 Dark · high contrast",
        })[value],
        index=0,
        key="analytics_chart_theme",
        help=(
            "A Project Statistics, Data Quality és Live Statistics Plotly-ábráinak háttér-/szövegkontrasztját vezérli."
            if language == "hu" else
            "Controls Plotly background/text contrast across Project Statistics, Data Quality and Live Statistics."
        ),
    )
    os.environ["TRAVEL_DATA_MODE"] = data_mode
    status = model_status()
    if status["exists"] and not status["stale"]:
        st.success("ML router: trained and current")
    elif status.get("reason") == "runtime_version_mismatch":
        st.warning("ML router: saved model was trained with another sklearn/Python version. Run setup once to retrain locally; package downgrade is not required.")
    elif status.get("reason") in {"model_metadata_missing", "model_load_failed"}:
        st.warning("ML router: legacy/incompatible saved artifact. Run setup once to retrain it safely in this environment.")
    elif status["exists"]:
        st.warning("ML router: model is stale after a dataset change")
    else:
        st.warning("ML router: not trained")
    st.caption("`local` is deterministic. `auto` may use live weather/FX providers with local fallback.")

chat_tab, tools_tab, quality_tab, train_tab, stats_tab, project_stats_tab, data_tab = st.tabs([
    "💬 Chat", "🛠 Tool Explorer", "🧪 Data Quality", "🧠 Train & Evaluate", "📈 Live Statistics", "📊 Project Statistics", "🗂 Dataset Explorer"
])

with chat_tab:
    st.subheader("Ask the travel agent")
    st.caption(f"Choose from {len(PRESET_QUESTIONS)} bilingual preset scenarios or write your own question.")

    if "chat_query" not in st.session_state:
        st.session_state.chat_query = ""
    if "loaded_preset_id" not in st.session_state:
        st.session_state.loaded_preset_id = None
    if "loaded_preset_text" not in st.session_state:
        st.session_state.loaded_preset_text = None

    with st.expander("📚 Preset question library", expanded=True):
        lang_key = "hu" if language == "hu" else "en"
        category_options = ["all"] + list(CATEGORY_LABELS)
        category_names = {"all": "Összes kategória" if lang_key == "hu" else "All categories"}
        category_names.update({key: labels[lang_key] for key, labels in CATEGORY_LABELS.items()})
        c1, c2 = st.columns([2, 1])
        selected_category = c1.selectbox(
            "Kategória" if lang_key == "hu" else "Category",
            category_options,
            format_func=lambda x: category_names[x],
            key="preset_category",
        )
        selected_difficulty = c2.selectbox(
            "Nehézség" if lang_key == "hu" else "Difficulty",
            ["all", "basic", "intermediate", "advanced"],
            format_func=lambda x: {
                "all": "Összes" if lang_key == "hu" else "All",
                "basic": "Alap" if lang_key == "hu" else "Basic",
                "intermediate": "Közepes" if lang_key == "hu" else "Intermediate",
                "advanced": "Haladó" if lang_key == "hu" else "Advanced",
            }[x],
            key="preset_difficulty",
        )
        filtered_presets = [
            item for item in PRESET_QUESTIONS
            if (selected_category == "all" or item["category"] == selected_category)
            and (selected_difficulty == "all" or item["difficulty"] == selected_difficulty)
        ]
        preset_map = {item["id"]: item for item in filtered_presets}
        selected_id = st.selectbox(
            "Előre megadott kérdés" if lang_key == "hu" else "Preset question",
            list(preset_map),
            format_func=lambda item_id: preset_text(preset_map[item_id], lang_key),
            key="preset_id",
        ) if preset_map else None
        if selected_id:
            chosen = preset_map[selected_id]
            st.caption(
                ("Elvárt toolok: " if lang_key == "hu" else "Expected tools: ")
                + " → ".join(chosen["expected_tools"])
            )
            load_col, count_col = st.columns([1, 3])
            if load_col.button("⬇ Kérdés betöltése" if lang_key == "hu" else "⬇ Load question", width="stretch"):
                loaded_text = preset_text(chosen, lang_key)
                st.session_state.chat_query = loaded_text
                st.session_state.loaded_preset_id = chosen["id"]
                st.session_state.loaded_preset_text = loaded_text
                st.rerun()
            count_col.caption(
                f"{len(filtered_presets)} / {len(PRESET_QUESTIONS)} "
                + ("kérdés látható a szűrés után." if lang_key == "hu" else "questions match the current filters.")
            )

    if language == "hu":
        st.info(
            "**Saját kérdést is írhatsz.** A legjobb eredményhez add meg a várost és azt, mit szeretnél: "
            "időjárás, hotel, étterem, látnivaló, közlekedés, deviza vagy költségszámítás. "
            "Ha fontos, írd bele az időtartamot, árkeretet, minimum értékelést vagy éttermi preferenciát is."
        )
        with st.expander("✍️ Hogyan írjak jó saját kérdést?"):
            st.markdown(
                """
- **Város:** pl. `Bécs`, `Budapest`, `Róma`
- **Időtartam:** pl. `3 napra megyek`
- **Keret:** pl. `150 euró alatti hotel`, `35 euró/fő alatti étterem`
- **Preferencia:** pl. `olasz étterem`, `múzeumok`, `minimum 4.5 értékelés`
- **Több kérés egy mondatban is lehet:** az agent több toolt is meghívhat.

**Példa:** `4 napra megyek Rómába. Nézd meg az időjárást, keress 180 euró alatti hotelt és ajánlj olasz éttermeket 40 euró/fő alatt.`
                """
            )
    else:
        st.info(
            "**You can write your own question.** For best results, include the city and the capabilities you need: "
            "weather, hotels, restaurants, attractions, transport, currency or calculations. Add duration, budget, rating or food preferences when relevant."
        )
        with st.expander("✍️ How to write a useful custom question"):
            st.markdown(
                """
- **City:** e.g. `Vienna`, `Budapest`, `Rome`
- **Duration:** e.g. `I am going for 3 days`
- **Budget:** e.g. `hotel under 150 EUR`, `restaurant under 35 EUR/person`
- **Preference:** e.g. `Italian food`, `museums`, `minimum 4.5 rating`
- **Combine multiple needs:** the agent may call several tools in one run.

**Example:** `I am going to Rome for 4 days. Check the weather, find a hotel under 180 EUR and recommend Italian restaurants under 40 EUR per person.`
                """
            )

    query = st.text_area(
        "Saját vagy betöltött kérdés" if language == "hu" else "Custom or loaded question",
        key="chat_query",
        height=150,
        placeholder="Írd ide a saját kérdésed természetes nyelven..." if language == "hu" else "Write your own travel question in natural language...",
    )
    col_run, col_clear = st.columns([1, 4])
    run_clicked = col_run.button("▶ Agent futtatása" if language == "hu" else "▶ Run agent", type="primary", width="stretch")
    def _clear_chat_query() -> None:
        st.session_state.chat_query = ""
        st.session_state.loaded_preset_id = None
        st.session_state.loaded_preset_text = None

    col_clear.button(
        "Törlés" if language == "hu" else "Clear",
        width="content",
        on_click=_clear_chat_query,
    )
    if run_clicked:
        if not query.strip():
            st.warning("Először adj meg egy kérdést." if language == "hu" else "Enter a question first.")
        elif methodology == "ml_router" and (not status["exists"] or status["stale"]):
            st.error("Az ML router modell hiányzik vagy elavult. Először tanítsd újra a Train & Evaluate fülön." if language == "hu" else "The ML router model is missing or stale. Open Train & Evaluate and train it first.")
        else:
            try:
                with st.spinner("Toolok tervezése és futtatása..." if language == "hu" else "Planning and executing tools..."):
                    agent = get_agent(methodology, language)
                    run = agent.run(query.strip())
                preset_matches = (
                    st.session_state.get("loaded_preset_id") is not None
                    and st.session_state.get("loaded_preset_text") == query.strip()
                )
                interaction_id = usage_store.record_run(
                    run,
                    methodology=methodology,
                    language=language,
                    data_mode=data_mode,
                    source="preset" if preset_matches else "custom",
                    preset_id=st.session_state.get("loaded_preset_id") if preset_matches else None,
                )
                st.markdown("### Válasz" if language == "hu" else "### Answer")
                st.write(run.answer)
                c1, c2, c3 = st.columns(3)
                c1.metric("Tool hívások" if language == "hu" else "Tool calls", len(run.trace))
                c2.metric("Teljes latency" if language == "hu" else "Total latency", f"{run.total_latency_ms:.1f} ms")
                c3.metric("Módszertan" if language == "hu" else "Methodology", run.metadata.get("methodology", methodology))
                st.caption(
                    (f"Helyi statisztikába mentve · futás #{interaction_id}" if language == "hu" else f"Saved to local statistics · run #{interaction_id}")
                )
                st.markdown("### Tool trace")
                show_trace(run)
                if run.metadata:
                    with st.expander("Agent metaadatok" if language == "hu" else "Agent metadata"):
                        st.json(run.metadata)
            except Exception as exc:
                st.exception(exc)

with tools_tab:
    st.subheader("Call tools directly")
    registry = ToolRegistry()
    tool = st.selectbox("Tool", registry.names)
    args: dict = {}
    if tool == "search_hotels":
        c1,c2,c3=st.columns(3); args["city"]=c1.text_input("City", "Vienna"); args["nights"]=c2.number_input("Nights",1,30,3); args["max_price_per_night_eur"]=c3.number_input("Max EUR/night",0.0,1000.0,150.0)
        c4,c5=st.columns(2); args["min_rating"]=c4.number_input("Min rating",0.0,5.0,4.0,0.1); args["top_k"]=c5.number_input("Top K",1,20,5)
    elif tool == "search_restaurants":
        c1,c2,c3=st.columns(3); args["city"]=c1.text_input("City", "Vienna"); cuisine=c2.text_input("Cuisine (optional)", ""); args["cuisines"]=[cuisine] if cuisine.strip() else None; args["max_meal_eur"]=c3.number_input("Max EUR/person",0.0,500.0,35.0)
        c4,c5,c6=st.columns(3); args["min_rating"]=c4.number_input("Min rating",0.0,5.0,4.0,0.1); args["vegetarian_only"]=c5.checkbox("Vegetarian only"); args["top_k"]=c6.number_input("Top K",1,20,6)
    elif tool == "search_attractions":
        c1,c2,c3=st.columns(3); args["city"]=c1.text_input("City", "Vienna"); category=c2.text_input("Category (optional)", "museum"); args["categories"]=[category] if category.strip() else None; max_ticket=c3.number_input("Max ticket EUR (0 = no limit)",0.0,200.0,30.0); args["max_ticket_eur"]=None if max_ticket==0 else max_ticket; args["top_k"]=st.number_input("Top K",1,20,6)
    elif tool == "get_weather":
        c1,c2=st.columns(2); args["city"]=c1.text_input("City", "Vienna"); args["days"]=c2.number_input("Days",1,14,3); args["unit"]="celsius"
    elif tool == "get_transport_options":
        c1,c2=st.columns(2); args["city"]=c1.text_input("City", "Vienna"); args["days"]=c2.number_input("Days",1,30,3)
    elif tool == "get_location_info":
        args["city"] = st.text_input("City", "Vienna")
    elif tool == "convert_currency":
        c1,c2,c3=st.columns(3); args["amount"]=c1.number_input("Amount",0.01,1000000.0,500.0); args["from_currency"]=c2.text_input("From", "EUR"); args["to_currency"]=c3.text_input("To", "HUF")
    elif tool == "calculate":
        args["expression"] = st.text_input("Expression", "120 * 3 + 75")
    st.code(json.dumps(args, ensure_ascii=False, indent=2), language="json")
    if st.button("Call tool", type="primary"):
        out,lat,ok,err=registry.execute(tool,args)
        (st.success if ok else st.error)(f"{'Success' if ok else err} · {lat:.2f} ms")
        st.json(out)

with quality_tab:
    def _rerun_quality_audit() -> None:
        with st.spinner("Auditing datasets..."):
            audit_all(write_outputs=True)
            load_quality_report.clear()
        st.success("Audit finished.")

    render_data_quality_dashboard(
        st,
        ROOT,
        load_quality_report(),
        language,
        rerun_callback=_rerun_quality_audit,
        chart_theme=chart_theme,
    )

with train_tab:
    st.subheader("Train and evaluate")
    st.caption("The full preparation action runs the same shared audit/training/evaluation modules used by the command-line pipeline.")
    if st.button("⚙ Full prepare: audit → train → 200-case check"):
        with st.spinner("Auditing data, training the router and running a smoke benchmark..."):
            prep_report = audit_all(write_outputs=True)
            if not all(prep_report["quality_gates"].values()):
                st.error("Data-quality gates failed. Training was stopped.")
            else:
                prep_metrics,_ = train_router(write_outputs=True)
                prep_agent = MLRouterTravelAgent(language=language)
                with tempfile.TemporaryDirectory(prefix="travel-agent-ui-smoke-") as tmpdir:
                    _, prep_summary = evaluate_agent(prep_agent, ROOT/"01_data/benchmark/agent_tasks.json", Path(tmpdir), 200)
                st.success("Preparation finished successfully.")
                c1,c2,c3=st.columns(3)
                c1.metric("Quality gates", f"{sum(prep_report['quality_gates'].values())}/{len(prep_report['quality_gates'])}")
                c2.metric("Held-out test F1", f"{prep_metrics['test_micro_f1']:.3f}")
                c3.metric("200-case tool selection", f"{prep_summary['tool_selection_accuracy']:.1%}")
                with st.expander("Preparation metrics"):
                    st.json({"training":prep_metrics,"agent_evaluation":prep_summary})
    st.divider()
    current = model_status()
    if current["exists"] and not current["stale"]:
        metrics=current.get("metrics",{})
        st.success("The saved router matches the current dataset.")
        c1,c2,c3=st.columns(3)
        c1.metric("Held-out test micro-F1", f"{metrics.get('test_micro_f1',0):.3f}")
        c2.metric("Challenge micro-F1", f"{metrics.get('challenge_micro_f1',0):.3f}")
        c3.metric("Training rows", f"{metrics.get('train_rows',0):,}")
    else:
        reason = current.get("reason", "missing_or_stale")
        if reason == "runtime_version_mismatch":
            st.warning("One-time retraining is required because the saved model was created by another sklearn/Python runtime. The installed packages can stay as they are.")
        elif reason in {"model_metadata_missing", "model_load_failed"}:
            st.warning("One-time retraining is required because the saved model artifact is legacy/incompatible.")
        else:
            st.warning("Training is recommended because the model is missing or stale.")
    if st.button("🧠 Train / retrain intent router", type="primary"):
        with st.spinner("Training word TF-IDF (1–3 grams) router..."):
            metrics,_=train_router(write_outputs=True)
        st.success("Training finished.")
        st.json(metrics)
    st.divider()
    eval_method=st.selectbox("Evaluation methodology",["plan_execute","ml_router"],index=1)
    eval_limit=st.select_slider("Cases",options=[50,100,250,500,1000,2500,7500,15000,22500],value=250)
    if st.button("Run evaluation"):
        if eval_method=="ml_router" and (not model_status()["exists"] or model_status()["stale"]):
            st.error("Train the ML router first.")
        else:
            with st.spinner(f"Running {eval_limit} benchmark cases..."):
                agent=get_agent(eval_method,language)
                out=ROOT/"06_results"/f"ui_{eval_method}"
                _,summary=evaluate_agent(agent,ROOT/"01_data/benchmark/agent_tasks.json",out,int(eval_limit))
            st.success("Evaluation finished.")
            metric_cols=st.columns(4)
            metric_cols[0].metric("Tool selection",f"{summary['tool_selection_accuracy']:.1%}")
            metric_cols[1].metric("Tool F1",f"{summary['tool_f1']:.1%}")
            metric_cols[2].metric("Argument accuracy",f"{summary['argument_accuracy']:.1%}")
            metric_cols[3].metric("Task success",f"{summary['task_success']:.1%}")
            st.json(summary)

with stats_tab:
    render_live_statistics_dashboard(st, usage_store, language, chart_theme=chart_theme)

with project_stats_tab:
    st.subheader("📊 Projektstatisztikák" if language == "hu" else "📊 Project statistics")
    st.caption(
        "Interaktív Plotly dashboard reprodukálható repository-, adat-, modell- és benchmark statisztikákkal."
        if language == "hu"
        else "Interactive Plotly dashboard with reproducible repository, data, model and benchmark statistics."
    )
    top_left, top_right = st.columns([1, 3])
    if top_left.button("🔄 Statisztikák újragenerálása" if language == "hu" else "🔄 Regenerate statistics", width="stretch"):
        with st.spinner("Statisztikák újraszámítása..." if language == "hu" else "Recomputing project statistics..."):
            subprocess.run([sys.executable, str(ROOT / "04_scripts" / "09_generate_project_statistics.py")], check=True)
            load_project_statistics.clear()
        st.success("Statisztikák frissítve." if language == "hu" else "Statistics refreshed.")
        st.rerun()
    top_right.info(
        "A dinamikus Plotly grafikonok a mentett CSV/JSON statisztikákból épülnek; a Saved Plots fülön a repositoryval együtt tárolt statikus PNG snapshotok is megmaradnak."
        if language == "hu"
        else "Interactive Plotly charts are built from saved CSV/JSON statistics; the Saved Plots tab also keeps the static PNG snapshots stored with the repository."
    )

    ps = load_project_statistics()
    if not ps:
        st.info(
            "Még nincs generált project statistics report. Kattints a fenti gombra."
            if language == "hu" else "No generated project statistics report yet. Use the button above."
        )
    else:
        render_dashboard(st, ROOT, ps, language, chart_theme=chart_theme)


with data_tab:
    st.subheader("Dataset explorer")
    dataset=st.selectbox("Dataset",["cities.csv","hotels.csv","attractions.csv","restaurants.csv","weather_fallback.csv","transport.csv","sample_user_queries.csv","intent_router_dataset.csv","intent_router_challenge.csv"])
    df=load_csv(dataset)
    c1,c2,c3=st.columns(3); c1.metric("Rows",f"{len(df):,}");c2.metric("Columns",len(df.columns));c3.metric("Missing cells",f"{int(df.isna().sum().sum()):,}")
    city_filter=None
    if "city" in df.columns:
        options=["All"]+sorted(df["city"].dropna().astype(str).unique().tolist())
        city_filter=st.selectbox("City filter",options)
    show=df if not city_filter or city_filter=="All" else df[df["city"].astype(str)==city_filter]
    st.dataframe(show.head(1000),width="stretch",hide_index=True)
    st.caption("Preview is limited to 1,000 rows; the underlying CSV remains unchanged.")

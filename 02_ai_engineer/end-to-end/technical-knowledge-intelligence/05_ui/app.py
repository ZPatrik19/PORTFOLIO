from __future__ import annotations

import hashlib
import time

import streamlit as st

from i18n import budget_description, chunk_description, normalize_language, pick, stage_label, tr
from tkip.config import PROJECT_ROOT, load_config
from tkip.gemini_service import GeminiService
from tkip.orchestration import KnowledgePlatform
from tkip.presets import ANSWER_PRESETS, CHUNK_PRESETS
from tkip.prompt_engineering import local_optimize
from ui_experiments import (
    _budget_label_map,
    _chunk_label_map,
    _profile_options,
    _request_from_settings,
    render_ab_results,
    render_benchmarking,
    render_prompt_preview,
)
from ui_pages import render_library, render_monitoring, render_workflow_page
from ui_rendering import render_answer, render_header


st.set_page_config(
    page_title="TKI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
  --tki-border: #334155;
  --tki-border-soft: #475569;
  --tki-soft: rgba(148,163,184,.12);
  --tki-card: rgba(255,255,255,.04);
  --tki-green: #22c55e;
  --tki-blue: #60a5fa;
  --tki-violet: #a78bfa;
  --tki-orange: #f59e0b;
  --tki-red: #ef4444;
}
.block-container {padding-top: 1.0rem; padding-bottom: 3rem; max-width: 1580px;}
[data-testid="stSidebar"] {min-width: 330px; max-width: 390px; border-right: 1px solid rgba(148,163,184,.22);}
[data-testid="stSidebar"] .block-container {padding-top: 1rem;}
.tki-hero {
  padding: 1.25rem 1.4rem; border: 1px solid rgba(148,163,184,.22); border-radius: 20px;
  background: linear-gradient(135deg, rgba(37,99,235,.18), rgba(15,23,42,.10), rgba(16,185,129,.12));
  margin-bottom: .9rem;
}
.tki-eyebrow {font-size:.76rem; font-weight:780; letter-spacing:.12em; opacity:.78; text-transform:uppercase;}
.tki-title {font-size:2.1rem; font-weight:820; margin:.18rem 0 .2rem 0; letter-spacing:-.025em;}
.tki-sub {opacity:.86; font-size:.98rem;}
.status-row {display:flex; gap:.45rem; flex-wrap:wrap; margin-top:.78rem;}
.status-pill {display:inline-flex; align-items:center; gap:.35rem; padding:.34rem .66rem; border-radius:999px; border:1px solid rgba(148,163,184,.24); font-size:.76rem; background:rgba(15,23,42,.22);}
.status-dot {width:.5rem; height:.5rem; border-radius:99px; display:inline-block; background:var(--tki-green); box-shadow:0 0 0 2px rgba(34,197,94,.14);}
.status-dot.off {background:var(--tki-orange); box-shadow:0 0 0 2px rgba(245,158,11,.14);}
.prompt-shell {border:1px solid rgba(148,163,184,.22); border-radius:18px; padding:1rem 1.05rem .85rem; background:rgba(15,23,42,.18); margin:.55rem 0 1rem;}
.section-kicker {font-size:.73rem; font-weight:780; letter-spacing:.09em; opacity:.72; text-transform:uppercase; margin-bottom:.25rem;}
.section-title {font-size:1.16rem; font-weight:780; margin-bottom:.42rem;}
.pipeline-flow {display:flex; align-items:stretch; gap:.25rem; overflow-x:auto; padding:.7rem .1rem 1rem .1rem;}
.pnode {min-width:138px; flex:1 0 138px; border:1px solid rgba(148,163,184,.24); border-radius:14px; padding:.72rem .72rem .62rem; background:rgba(15,23,42,.18); position:relative;}
.pnode:after {content:'→'; position:absolute; right:-.33rem; top:39%; opacity:.55; font-weight:900; z-index:3;}
.pnode:last-child:after {display:none;}
.pnode.success {border-top:4px solid var(--tki-green);}
.pnode.warning {border-top:4px solid var(--tki-orange);}
.pnode.fallback {border-top:4px solid var(--tki-violet);}
.pnode.skipped {border-top:4px solid #94a3b8; opacity:.88;}
.pnode.blocked {border-top:4px solid var(--tki-red);}
.picon {font-size:1.05rem; margin-bottom:.25rem;}
.pname {font-size:.80rem; font-weight:780; line-height:1.18;}
.ptime {font-size:.94rem; font-weight:780; margin-top:.35rem;}
.pdetail {font-size:.70rem; opacity:.78; line-height:1.25; margin-top:.22rem; max-height:2.8em; overflow:hidden;}
.evidence-card {border:1px solid rgba(148,163,184,.24); border-radius:14px; padding:.82rem .9rem; margin:.5rem 0; background:rgba(15,23,42,.18);}
.evidence-title {font-size:.92rem; font-weight:780; margin-bottom:.18rem;}
.evidence-meta {font-size:.74rem; opacity:.78; margin-bottom:.45rem;}
.evidence-text {font-size:.89rem; line-height:1.48;}
.tool-card {border-left:4px solid var(--tki-violet); border-radius:10px; padding:.64rem .78rem; background:rgba(139,92,246,.08); margin:.45rem 0;}
.qa-good {border-left:4px solid var(--tki-green); padding:.58rem .72rem; background:rgba(34,197,94,.08); border-radius:10px;}
.qa-warn {border-left:4px solid var(--tki-orange); padding:.58rem .72rem; background:rgba(245,158,11,.08); border-radius:10px;}
.small-muted {opacity:.72; font-size:.81rem;}
.code-caption {font-size:.72rem; opacity:.72; margin-top:-.2rem;}
[data-testid="stMetricValue"] {font-weight:800;}
</style>
""",
    unsafe_allow_html=True,
)

QUESTIONS = {
    "hu": [
        "Magyarázd el a RAG működését a könyveim alapján, és mutasd meg a fő komponenseket.",
        "Hasonlítsd össze, hogyan magyarázzák a könyveim a Transformer attention mechanizmust.",
        "Mi az ETL, melyek a fő lépései, és milyen gyakori hibák fordulnak elő?",
        "Keress PyTorch Dataset és DataLoader implementációs mintákat a könyveimben.",
        "Tanítsd meg nekem az embeddingek működését intuíció → matematika → kód sorrendben.",
        "Mit írnak a könyveim az LLM/RAG rendszerek production monitoringjáról?",
        "Mi a különbség a BM25, dense retrieval és hybrid retrieval között?",
        "Mikor érdemes rerankert használni egy RAG rendszerben, és milyen trade-offjai vannak?",
        "Hogyan működik a semantic chunking, és mikor jobb a structure-aware chunkingnál?",
        "Magyarázd el a vector database szerepét egy production RAG rendszerben.",
        "Milyen tipikus hallucination problémák vannak LLM rendszerekben, és hogyan mérhetők?",
        "Mutasd be a tool calling teljes workflow-ját egy AI assistant rendszerben.",
        "Hogyan terveznél FastAPI backendet egy production AI/RAG szolgáltatáshoz?",
        "Mit írnak a könyveim a Docker networking működéséről?",
        "Magyarázd el a Kubernetes Deployment és Service kapcsolatát.",
        "Hasonlítsd össze a supervised és unsupervised learning legfontosabb felhasználási eseteit.",
        "Magyarázd el a Random Forest működését és a legfontosabb hyperparamétereket.",
        "Tanítsd meg a PCA-t intuíció → matematika → Python implementáció sorrendben.",
        "Milyen production problémákat okozhat data drift és concept drift?",
        "Mi a különbség ETL és ELT között, és mikor melyiket érdemes választani?",
        "Hogyan működik a Databricks Lakehouse architektúra a könyveim alapján?",
        "Mutasd be az adversarial machine learning legfontosabb támadási és védekezési módszereit.",
        "Milyen monitoring metrikákat érdemes mérni egy LLM/RAG rendszerben?",
        "Készíts tanulási tervet LLM Engineering témából a könyveim alapján.",
        "Hasonlítsd össze a LangChain és a FastAPI szerepét egy production AI rendszerben.",
        "Mikor érdemes több chunking stratégiát külön indexekben fenntartani?",
        "Magyarázd el, hogyan működik a prompt injection elleni védekezés egy RAG rendszerben.",
        "Adj példát arra, hogyan nézne ki egy strukturált JSON kimenet egy AI asszisztensnél.",
        "Hasonlítsd össze a fine-tuning, a RAG és a tool calling szerepét üzleti környezetben.",
        "Mutasd be, hogyan építenél benchmarking rendszert szövegosztályozásra, clusteringre és topic modelingra.",
        "Milyen lépésekből áll egy production LLM alkalmazás hibaanalízise és regressziós tesztelése?",
        "Készíts összehasonlítást a különböző embedding-modellek használatáról retrieval célra.",
        "Tanítsd meg a rendszertervezési kompromisszumokat: gyorsaság, költség, minőség egy RAG pipeline-ban.",
        "Adj egy end-to-end magyarázatot arról, hogyan lesz a dokumentumokból válasz a teljes rendszerben.",
        "Hasonlítsd össze a LangChain RunnableSequence, RunnableParallel és ToolNode mintákat egy RAG workflow-ban.",
        "Mutasd meg, hogyan nézne ki egy LangGraph state machine kutató asszisztenshez.",
        "Milyen metrikákkal hasonlítanád össze a BM25, dense, hybrid és reranking retrievalt?",
        "Hogyan mérnéd a promptprofilok minőségét, költségét és késleltetését?",
        "Mikor tekinthető egy RAG válasz faithfulness szempontból jónak?",
        "Milyen teszteseteket írnál tool calling argumentumhibákra és jogosulatlan eszközhívásra?",
        "Hogyan építenél golden setet RAG regressziós teszteléshez?",
        "Magyarázd el, hogyan lehet citation correctness és citation completeness metrikát mérni.",
        "Mutass egy production-ready Pydantic sémát strukturált LLM válaszhoz.",
        "Milyen hibák fordulhatnak elő embedding cache, index rebuild és metadata filter használatakor?",
    ],
    "en": [
        "Explain RAG using my books and show the main system components.",
        "Compare how my books explain the Transformer attention mechanism.",
        "What is ETL, what are its main stages, and what common failures should I know?",
        "Find PyTorch Dataset and DataLoader implementation patterns in my books.",
        "Teach me embeddings in the order intuition → mathematics → code.",
        "What do my books say about production monitoring for LLM/RAG systems?",
        "Explain the differences between BM25, dense retrieval and hybrid retrieval.",
        "When should a RAG system use a reranker and what are the trade-offs?",
        "How does semantic chunking work and when is it better than structure-aware chunking?",
        "Explain the role of a vector database in a production RAG architecture.",
        "What hallucination failure modes occur in LLM systems and how can they be measured?",
        "Show the complete tool-calling workflow of an AI assistant.",
        "How would you design a FastAPI backend for a production AI/RAG service?",
        "What do my books say about Docker networking?",
        "Explain the relationship between Kubernetes Deployment and Service.",
        "Compare the most important use cases of supervised and unsupervised learning.",
        "Explain Random Forest and its most important hyperparameters.",
        "Teach me PCA in the order intuition → mathematics → Python implementation.",
        "What production problems can data drift and concept drift cause?",
        "What is the difference between ETL and ELT and when should each be used?",
        "Explain the Databricks Lakehouse architecture using my books.",
        "Summarize the main adversarial machine-learning attack and defense methods.",
        "Which monitoring metrics should a production LLM/RAG system track?",
        "Create a learning path for LLM Engineering from my library.",
        "Compare the roles of LangChain and FastAPI in a production AI system.",
        "When should you maintain multiple chunking strategies as separate indexes?",
        "Explain how prompt-injection defenses work in a RAG system.",
        "Give an example of a structured JSON output for an AI assistant.",
        "Compare fine-tuning, RAG and tool calling in a business environment.",
        "Show how you would benchmark text classification, clustering and topic modeling.",
        "What steps belong in failure analysis and regression testing for a production LLM application?",
        "Compare different embedding-model choices for retrieval.",
        "Teach the system-design trade-offs between speed, cost and answer quality in a RAG pipeline.",
        "Give an end-to-end explanation of how documents become answers in the full system.",
        "Compare LangChain RunnableSequence, RunnableParallel and ToolNode patterns in a RAG workflow.",
        "Show how a LangGraph-style state machine could model a research assistant.",
        "Which metrics would you use to compare BM25, dense, hybrid and reranked retrieval?",
        "How would you measure prompt-profile quality, cost and latency?",
        "When should a RAG answer be considered good in terms of faithfulness?",
        "Which test cases would you write for tool-call argument errors and unauthorized tools?",
        "How would you build a golden set for RAG regression testing?",
        "Explain how citation correctness and citation completeness can be measured.",
        "Show a production-ready Pydantic schema for structured LLM output.",
        "What failures can occur with embedding caches, index rebuilds and metadata filtering?",
    ],
}

@st.cache_resource(show_spinner=False)
def get_platform() -> KnowledgePlatform:
    platform = KnowledgePlatform(load_config())
    platform.ensure_ready()
    return platform


def _source_label(language: str, value: str) -> str:
    labels = {
        "all": pick(language, "Összes", "All"),
        "private": pick(language, "Saját könyvtár", "User library"),
        "public": pick(language, "Referenciaanyag", "Reference docs"),
        "demo": pick(language, "Demo", "Demo"),
    }
    return labels.get(value, value)


LANGUAGE_BOUND_SESSION_KEYS = (
    "workspace_question",
    "last_answer",
    "last_answer_new",
    "previous_answer",
    "ab_question",
    "ab_results",
    "prompt_benchmark_df",
)


def _language_selector() -> str:
    """Select the UI language and repair stale legacy Streamlit session values."""

    current = normalize_language(
        st.session_state.get("ui_language_selector", st.session_state.get("ui_lang", "hu"))
    )
    # Streamlit validates widget state against the raw option values.  Older
    # releases stored labels such as ``Angol``; remove them before creating the
    # code-based selector so a stale browser session cannot leak into QUESTIONS.
    raw_widget_value = st.session_state.get("ui_language_selector")
    if raw_widget_value not in {"hu", "en", None}:
        st.session_state.pop("ui_language_selector", None)

    selected = st.sidebar.selectbox(
        tr(current, "language"),
        ["hu", "en"],
        index=0 if current == "hu" else 1,
        key="ui_language_selector",
        format_func=lambda code: tr(current, "hungarian" if code == "hu" else "english"),
    )
    selected = normalize_language(selected)
    previous = normalize_language(st.session_state.get("_ui_lang_applied", selected))
    if selected != previous:
        for key in LANGUAGE_BOUND_SESSION_KEYS:
            st.session_state.pop(key, None)
    st.session_state["ui_lang"] = selected
    st.session_state["_ui_lang_applied"] = selected
    return selected


ui_lang = _language_selector()

try:
    kp = get_platform()
except Exception as exc:
    st.error(f"{tr(ui_lang, 'index_not_ready')}: {exc}")
    st.code(tr(ui_lang, "run_setup"), language="text")
    st.stop()

if "gemini_key" not in st.session_state:
    st.session_state["gemini_key"] = ""

st.sidebar.markdown(f"### {tr(ui_lang, 'api_section')}")
entered_key = st.sidebar.text_input(
    tr(ui_lang, "api_key"),
    type="password",
    key="gemini_key",
    placeholder=tr(ui_lang, "api_key_placeholder"),
    help=tr(ui_lang, "api_help"),
)
effective_key = entered_key.strip() or None
gem_probe = GeminiService(kp.cfg, api_key=effective_key)
gemini_client_ready = gem_probe.available
key_material = gem_probe.key or ""
key_fingerprint = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:12] if key_material else "none"
if st.session_state.get("gemini_verified_fp") != key_fingerprint:
    st.session_state["gemini_verified"] = False
    st.session_state["gemini_verified_fp"] = key_fingerprint
    st.session_state["gemini_connection_message"] = ""

gemini_available = bool(gemini_client_ready and st.session_state.get("gemini_verified"))
if gemini_available:
    st.sidebar.success(tr(ui_lang, "gemini_verified"))
elif gemini_client_ready:
    st.sidebar.info(tr(ui_lang, "gemini_loaded"))
else:
    detail = gem_probe.init_error or ""
    st.sidebar.warning(f"{tr(ui_lang, 'gemini_unavailable')}{': ' + detail if detail else ''}")

if st.sidebar.button(tr(ui_lang, "test_connection"), use_container_width=True, disabled=not gemini_client_ready):
    with st.sidebar.status(tr(ui_lang, "testing_connection"), expanded=True) as status:
        ok, message = gem_probe.test_connection()
        st.session_state["gemini_verified"] = bool(ok)
        st.session_state["gemini_verified_fp"] = key_fingerprint
        st.session_state["gemini_connection_message"] = message
        gemini_available = bool(ok)
        if ok:
            status.update(label=tr(ui_lang, "gemini_verified"), state="complete", expanded=False)
        else:
            status.update(label=tr(ui_lang, "gemini_connection_failed"), state="error", expanded=True)
            st.sidebar.error(message)

if effective_key and st.sidebar.button(tr(ui_lang, "save_key"), use_container_width=True, help=tr(ui_lang, "api_help")):
    env_path = PROJECT_ROOT / ".env"
    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    lines = [line for line in existing.splitlines() if not line.strip().startswith("GEMINI_API_KEY=")]
    lines.append(f"GEMINI_API_KEY={effective_key}")
    env_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    st.sidebar.success(tr(ui_lang, "saved_key"))

st.sidebar.divider()
nav_items = [
    tr(ui_lang, "workspace"),
    tr(ui_lang, "workflow"),
    tr(ui_lang, "benchmarking"),
    tr(ui_lang, "playground"),
    tr(ui_lang, "library"),
    tr(ui_lang, "monitoring"),
]
nav = st.sidebar.radio(tr(ui_lang, "navigation"), nav_items, index=0)
render_header(kp, ui_lang, gemini_available)

if nav == tr(ui_lang, "workspace"):
    qs = QUESTIONS[normalize_language(ui_lang)]
    selected_example = st.selectbox(
        tr(ui_lang, "question_bank"),
        ["—"] + qs,
        index=0,
        help=pick(ui_lang, "Gyors próbakérdések a rendszer teszteléséhez.", "Quick test questions for the system."),
    )
    if selected_example != "—" and st.button(tr(ui_lang, "use_question"), use_container_width=True):
        st.session_state["workspace_question"] = selected_example

    if "workspace_question" not in st.session_state:
        st.session_state["workspace_question"] = ""
    question = st.text_area(
        tr(ui_lang, "question"),
        key="workspace_question",
        height=135,
        placeholder=qs[0],
        help=tr(ui_lang, "question_help"),
    )

    st.markdown(f"### {tr(ui_lang, 'answer_settings')}")

    st.markdown(f"#### 1 · {tr(ui_lang, 'answer_budget')}")
    budget_labels = _budget_label_map(ui_lang)
    budget_label = st.radio(
        tr(ui_lang, "answer_budget"),
        list(budget_labels),
        index=1,
        horizontal=True,
        label_visibility="collapsed",
    )
    budget_key = budget_labels[budget_label]
    budget = ANSWER_PRESETS[budget_key]
    b1, b2, b3, b4 = st.columns(4)
    b1.metric(tr(ui_lang, "context_budget"), f"{budget['context_max_chars']:,}", help=budget_description(ui_lang, budget_key, budget["description"]))
    b2.metric(tr(ui_lang, "output_budget"), f"{budget['max_output_tokens']:,}")
    b3.metric(tr(ui_lang, "top_evidence"), budget["retrieval_final_k"])
    b4.metric(tr(ui_lang, "max_tools"), budget["max_tool_calls"])

    st.markdown(f"#### 2 · {tr(ui_lang, 'prompt_profile')}")
    profile_map = _profile_options(ui_lang)
    default_profile_label = next(label for label, key in profile_map.items() if key == budget["prompt_profile"])
    prompt_label = st.selectbox(tr(ui_lang, "prompt_profile"), list(profile_map), index=list(profile_map).index(default_profile_label), label_visibility="collapsed")
    prompt_profile = profile_map[prompt_label]
    with st.expander(tr(ui_lang, "prompt_preview"), expanded=False):
        render_prompt_preview(question, prompt_profile, ui_lang)
    st.markdown(f"#### 3 · {tr(ui_lang, 'chunk_profile')}")
    chunk_map = _chunk_label_map(ui_lang)
    chunk_labels = list(chunk_map)
    chunk_label = st.radio(
        tr(ui_lang, "chunk_profile"),
        chunk_labels,
        index=1,
        horizontal=True,
        label_visibility="collapsed",
    )
    chunk_key = chunk_map[chunk_label]
    chunk = CHUNK_PRESETS[chunk_key]
    c1, c2, c3 = st.columns(3)
    c1.metric(tr(ui_lang, "persistent_index"), chunk["index_variant"], help=chunk_description(ui_lang, chunk_key, chunk["description"]))
    c2.metric(tr(ui_lang, "chunk_target"), f"{chunk['chunk_size']}")
    c3.metric(tr(ui_lang, "overlap"), chunk["overlap"])

    prompt_check = True
    quality_review = bool(budget["quality_review"])
    custom_instruction = None
    include_visual = True
    source_type = "all"
    chunk_type = "all"
    topic = ""
    selected_docs: list[str] = []
    gemini_optimize = True

    with st.expander(tr(ui_lang, "advanced_controls"), expanded=False):
        x1, x2 = st.columns(2)
        prompt_check = x1.toggle(tr(ui_lang, "prompt_check"), value=True)
        quality_review = x2.toggle(tr(ui_lang, "quality_review"), value=quality_review)
        y1, y2 = st.columns(2)
        gemini_optimize = y1.toggle(tr(ui_lang, "gemini_prompt_refine"), value=True, disabled=not gemini_client_ready)
        include_visual = y2.toggle(tr(ui_lang, "source_visual"), value=True)
        custom_instruction = st.text_area(
            tr(ui_lang, "custom_instruction"),
            height=80,
            placeholder=tr(ui_lang, "custom_instruction_placeholder"),
        )
        st.markdown(f"**{tr(ui_lang, 'corpus_filters')}**")
        f1, f2, f3 = st.columns(3)
        doc_titles = sorted({chunk.title for chunk in kp.chunks})
        selected_titles = f1.multiselect(tr(ui_lang, "documents_filter"), doc_titles)
        selected_docs = list(dict.fromkeys(chunk.document_id for chunk in kp.chunks if chunk.title in selected_titles))
        source_type = f2.selectbox(
            tr(ui_lang, "source_filter"),
            ["all", "private", "public", "demo"],
            format_func=lambda value: _source_label(ui_lang, value),
        )
        chunk_types = sorted({chunk.chunk_type for chunk in kp.chunks})
        chunk_type = f3.selectbox(
            tr(ui_lang, "chunk_type_filter"),
            ["all"] + chunk_types,
            format_func=lambda value: tr(ui_lang, "all") if value == "all" else value,
        )
        topic = st.text_input(tr(ui_lang, "topic_filter"), value="")

    if not gemini_client_ready:
        st.warning(pick(ui_lang, "Adj meg Gemini API-kulcsot az oldalsávban.", "Add a Gemini API key in the sidebar."))
    run = st.button(
        f"✨ {tr(ui_lang, 'generate')}",
        type="primary",
        use_container_width=True,
        disabled=(not bool(question.strip()) or not gemini_client_ready),
    )
    if run:
        connection_ok = gemini_available
        connection_message = st.session_state.get("gemini_connection_message", "")
        if not connection_ok:
            with st.spinner(tr(ui_lang, "validating_gemini")):
                connection_ok, connection_message = gem_probe.test_connection()
            st.session_state["gemini_verified"] = bool(connection_ok)
            st.session_state["gemini_verified_fp"] = key_fingerprint
            st.session_state["gemini_connection_message"] = connection_message
        if not connection_ok:
            st.error(f"{tr(ui_lang, 'gemini_validation_failed')}\n\n{connection_message}")
        else:
            request = _request_from_settings(
                question=question,
                language=ui_lang,
                budget_key=budget_key,
                prompt_profile=prompt_profile,
                chunk_key=chunk_key,
                prompt_check=prompt_check,
                quality_review=quality_review,
                custom_instruction=custom_instruction,
                include_source_visual=include_visual,
                gemini_optimize_prompt=gemini_optimize,
                selected_docs=selected_docs,
                source_type=None if source_type == "all" else source_type,
                chunk_type=None if chunk_type == "all" else chunk_type,
                topic=topic or None,
            )
            progress = st.progress(0, text=tr(ui_lang, "starting_pipeline"))
            live_box = st.empty()

            def on_stage(stage: dict, pipeline: list[dict]) -> None:
                pct = min(95, int(100 * len(pipeline) / 19))
                progress.progress(
                    pct,
                    text=stage_label(ui_lang, str(stage.get("stage") or "")),
                )
                live_box.caption(
                    " → ".join(
                        stage_label(ui_lang, str(item.get("stage") or ""))
                        for item in pipeline[-6:]
                    )
                )

            try:
                answer = kp.ask(request, gemini_api_key=effective_key, progress_callback=on_stage)
                progress.progress(100, text=tr(ui_lang, "answer_ready"))
                time.sleep(0.1)
                progress.empty()
                live_box.empty()
                st.session_state["previous_answer"] = st.session_state.get("last_answer")
                st.session_state["last_answer"] = answer
                st.session_state["last_answer_new"] = True
            except Exception as exc:
                progress.empty()
                live_box.empty()
                st.error(f"{tr(ui_lang, 'pipeline_failed')}\n\n{exc}")

    if "last_answer" in st.session_state:
        is_new = bool(st.session_state.get("last_answer_new", False))
        render_answer(
            st.session_state["last_answer"],
            ui_lang,
            is_new,
            kp,
            st.session_state.get("previous_answer"),
        )
        st.session_state["last_answer_new"] = False

elif nav == tr(ui_lang, "workflow"):
    render_workflow_page(kp, ui_lang)

elif nav == tr(ui_lang, "benchmarking"):
    render_benchmarking(kp, ui_lang, effective_key, gemini_available, QUESTIONS)

elif nav == tr(ui_lang, "playground"):
    st.markdown(f"## {tr(ui_lang, 'ab_title')}")
    if not gemini_available:
        st.warning(tr(ui_lang, "ab_requires_gemini"))
    if "ab_question" not in st.session_state:
        st.session_state["ab_question"] = QUESTIONS[normalize_language(ui_lang)][0]
    ab_question = st.text_area(tr(ui_lang, "ab_question"), key="ab_question", height=120)

    profile_map = _profile_options(ui_lang)
    budget_map = _budget_label_map(ui_lang)
    chunk_map = _chunk_label_map(ui_lang)
    column_a, column_b = st.columns(2)
    configs: dict[str, dict] = {}
    for column, key, default_budget, default_profile, default_chunk in [
        (column_a, "A", "economy", "concise_expert", "compact"),
        (column_b, "B", "deep", "technical_deep_dive", "semantic_deep"),
    ]:
        with column:
            st.markdown(f"### {tr(ui_lang, 'variant')} {key}")
            budget_default_label = next(label for label, value in budget_map.items() if value == default_budget)
            budget_label_value = st.selectbox(
                f"{tr(ui_lang, 'answer_budget')} {key}",
                list(budget_map),
                index=list(budget_map).index(budget_default_label),
                key=f"ab_budget_{key}",
            )
            budget_key_value = budget_map[budget_label_value]
            base = ANSWER_PRESETS[budget_key_value]

            profile_default_label = next(label for label, value in profile_map.items() if value == default_profile)
            profile_label_value = st.selectbox(
                f"{tr(ui_lang, 'prompt_profile')} {key}",
                list(profile_map),
                index=list(profile_map).index(profile_default_label),
                key=f"ab_prompt_{key}",
            )
            profile_key_value = profile_map[profile_label_value]

            chunk_default_label = next(label for label, value in chunk_map.items() if value == default_chunk)
            chunk_label_value = st.selectbox(
                f"{tr(ui_lang, 'chunk_profile')} {key}",
                list(chunk_map),
                index=list(chunk_map).index(chunk_default_label),
                key=f"ab_chunk_{key}",
            )
            chunk_key_value = chunk_map[chunk_label_value]

            with st.expander(f"{tr(ui_lang, 'expert_controls')} {key}", expanded=False):
                context_chars = st.slider(f"{tr(ui_lang, 'context_chars')} {key}", 6000, 40000, int(base["context_max_chars"]), 1000, key=f"ab_ctx_{key}")
                output_tokens = st.slider(f"{tr(ui_lang, 'max_output_tokens')} {key}", 500, 6000, int(base["max_output_tokens"]), 250, key=f"ab_out_{key}")
                tools = st.slider(f"{tr(ui_lang, 'max_tool_calls')} {key}", 0, 6, int(base["max_tool_calls"]), 1, key=f"ab_tools_{key}")
                final_k = st.slider(f"{tr(ui_lang, 'final_evidence_chunks')} {key}", 3, 15, int(base["retrieval_final_k"]), 1, key=f"ab_k_{key}")
                temperature = st.slider(f"{tr(ui_lang, 'temperature')} {key}", 0.0, 1.5, float(base["temperature"]), 0.05, key=f"ab_temp_{key}")
                runtime = st.toggle(f"{tr(ui_lang, 'query_time_rechunk')} {key}", value=False, key=f"ab_rt_{key}")

            configs[key] = {
                "budget_key": budget_key_value,
                "prompt_profile": profile_key_value,
                "chunk_key": chunk_key_value,
                "context_chars": context_chars,
                "output_tokens": output_tokens,
                "tools": tools,
                "final_k": final_k,
                "temp": temperature,
                "runtime": runtime,
            }
            with st.expander(tr(ui_lang, "prompt_preview"), expanded=False):
                preview = local_optimize(ab_question, profile_key_value, ui_lang)
                st.code(preview["optimized"][:2800], language="text")

    run_ab = st.button(
        f"⚖️ {tr(ui_lang, 'run_ab')}",
        type="primary",
        use_container_width=True,
        disabled=(not gemini_available or not ab_question.strip()),
    )
    if run_ab:
        results = {}
        for key in ["A", "B"]:
            config = configs[key]
            request = _request_from_settings(
                question=ab_question,
                language=ui_lang,
                budget_key=config["budget_key"],
                prompt_profile=config["prompt_profile"],
                chunk_key=config["chunk_key"],
                context_max_chars=config["context_chars"],
                max_output_tokens=config["output_tokens"],
                max_tool_calls=config["tools"],
                retrieval_final_k=config["final_k"],
                temperature=config["temp"],
                runtime_chunking=config["runtime"],
                include_source_visual=False,
                quality_review=True,
                gemini_optimize_prompt=True,
            )
            with st.status(f"{tr(ui_lang, 'running_variant')} {key}…", expanded=False) as status:
                results[key] = kp.ask(request, gemini_api_key=effective_key)
                status.update(label=f"{tr(ui_lang, 'variant_ready')} {key}", state="complete")
        st.session_state["ab_results"] = results
    if "ab_results" in st.session_state:
        render_ab_results(st.session_state["ab_results"]["A"], st.session_state["ab_results"]["B"], ui_lang)

elif nav == tr(ui_lang, "library"):
    render_library(kp, ui_lang)

else:
    render_monitoring(kp, ui_lang)

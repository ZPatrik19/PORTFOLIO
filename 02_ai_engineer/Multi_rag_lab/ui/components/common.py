from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

from rag_engine.platform.config import load_settings
from rag_engine.presets import CHUNKING_STRATEGIES, CONTEXT_PROFILES, PROMPT_PROFILES, RAG_STRATEGIES
from rag_engine.indexing.embedding_factory import create_embedding_provider
from rag_engine.platform.profiles import load_llm_profiles, resolve_profile_name
from rag_engine.models import ChunkingConfig
from rag_engine.retrieval.rerank_cross_encoder import CrossEncoderReranker
from rag_engine.service import build_retrieval_resources, compose_lab
from rag_engine.indexing.prebuilt import DEFAULT_STATE as PREBUILT_INDEX_STATE


EMBEDDING_LABELS = {
    "multilingual": "Multilingual MiniLM (magyar baseline)",
    "e5-small": "Multilingual E5 Small (query/passage prefix)",
    "english": "Angol Sentence Transformer",
    "hashing": "Hashing fallback (offline/CI)",
}


def ensure_state() -> None:
    settings = load_settings()
    medical = sorted((ROOT / "data" / "raw" / "hungarian_medical").glob("*.html"))
    default_paths = [str(path) for path in medical] or [str(ROOT / "data" / "demo" / "magyar_rag_demo.md")]
    defaults = {
        "document_paths": default_paths,
        "chunking_strategy": "recursive",
        "chunk_size": 500,
        "chunk_overlap": 100,
        "semantic_threshold": 0.72,
        "embedding_mode": "e5-small",
        "embedding_device": "auto",
        "vector_device": "cpu",
        "retrieval_mode": "hybrid",
        "fusion_mode": "rrf",
        "rrf_k": 60,
        "dense_weight": 0.5,
        "top_k": 5,
        "candidate_count": 20,
        "reranker_mode": "lexical",
        "reranker_device": "auto",
        "rag_strategy": "hybrid",
        "context_budget": 1800,
        "llm_provider": settings.llm_provider,
        "ollama_profile": settings.ollama_profile,
        "faiss_profile": "cpu",
        "cuda_profile": "auto",
        "context_profile": "balanced",
        "prompt_profile": "professional",
        "presentation_mode": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    apply_pending_widget_state_updates()


def queue_widget_state_updates(**updates: object) -> None:
    """Schedule shared widget-state changes for the next Streamlit rerun.

    Shared controls live in the sidebar and are instantiated before individual
    pages render. Streamlit forbids mutating a widget-owned session-state key
    after that widget has been created in the same run. Page actions therefore
    queue changes under a private key and rerun; ``ensure_state`` applies them
    before any widgets are instantiated.
    """
    pending = dict(st.session_state.get("_pending_widget_state_updates", {}))
    pending.update(updates)
    st.session_state["_pending_widget_state_updates"] = pending


def apply_pending_widget_state_updates() -> None:
    pending = st.session_state.pop("_pending_widget_state_updates", {})
    if not isinstance(pending, dict):
        return
    for key, value in pending.items():
        st.session_state[key] = value


def active_paths() -> list[Path]:
    return [Path(p) for p in st.session_state.get("document_paths", []) if Path(p).exists()]


def selected_embedding_model() -> tuple[str, bool]:
    settings = load_settings()
    mode = st.session_state.get("embedding_mode", "multilingual")
    if mode == "hashing":
        return "__offline_hashing_fallback__", True
    if mode == "multilingual":
        return settings.multilingual_embedding_model, False
    if mode == "e5-small":
        return settings.e5_embedding_model, False
    return settings.embedding_model, False


@st.cache_resource(show_spinner="Beágyazási modell betöltése...")
def cached_embedding_provider(model_name: str, device: str, fallback_embedding: bool):
    return create_embedding_provider(
        model_name,
        device=device,
        fallback_to_hashing=fallback_embedding,
    )


@st.cache_resource(show_spinner="Dokumentumindex és retrieval erőforrások betöltése...")
def cached_retrieval_resources(
    path_signatures: tuple[tuple[str, int], ...],
    index_state_mtime: int,
    strategy: str,
    chunk_size: int,
    overlap: int,
    semantic_threshold: float,
    embedding_model: str,
    embedding_device: str,
    vector_device: str,
    fallback_embedding: bool,
):
    del index_state_mtime  # cache-key only: invalidates resources after a persisted reindex
    paths = [Path(item[0]) for item in path_signatures]
    return build_retrieval_resources(
        paths,
        chunking=ChunkingConfig(
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            semantic_threshold=semantic_threshold,
        ),
        embedding_model=embedding_model,
        embedding_device=embedding_device,
        vector_device=vector_device,
        fallback_embedding=fallback_embedding,
    )


@st.cache_resource(show_spinner="RAG runtime összeállítása...")
def cached_lab(
    path_signatures: tuple[tuple[str, int], ...],
    index_state_mtime: int,
    strategy: str,
    chunk_size: int,
    overlap: int,
    semantic_threshold: float,
    embedding_model: str,
    embedding_device: str,
    vector_device: str,
    llm_provider: str,
    ollama_model: str,
    fallback_embedding: bool,
    fusion_mode: str,
    rrf_k: int,
    dense_weight: float,
    ollama_connect_timeout: float,
    ollama_read_timeout: float,
    ollama_max_retries: int,
    ollama_num_predict: int,
    ollama_temperature: float,
    ollama_keep_alive: str,
):
    settings = load_settings()
    resources = cached_retrieval_resources(
        path_signatures,
        index_state_mtime,
        strategy,
        chunk_size,
        overlap,
        semantic_threshold,
        embedding_model,
        embedding_device,
        vector_device,
        fallback_embedding,
    )
    return compose_lab(
        resources,
        llm_provider=llm_provider,
        ollama_base_url=settings.ollama_base_url,
        ollama_model=ollama_model,
        fusion=fusion_mode,
        rrf_k=rrf_k,
        dense_weight=dense_weight,
        ollama_connect_timeout=ollama_connect_timeout,
        ollama_read_timeout=ollama_read_timeout,
        ollama_max_retries=ollama_max_retries,
        ollama_num_predict=ollama_num_predict,
        ollama_temperature=ollama_temperature,
        ollama_keep_alive=ollama_keep_alive,
    )


def get_lab(
    *,
    strategy: str | None = None,
    embedding_device: str | None = None,
    vector_device: str | None = None,
):
    settings = load_settings()
    paths = active_paths()
    if not paths:
        raise RuntimeError("Nincs aktív dokumentum. Tölts le vagy tölts fel dokumentumot a Dokumentumok oldalon.")
    signatures = tuple((str(p), p.stat().st_mtime_ns) for p in paths)
    model, fallback_embedding = selected_embedding_model()
    llm_profiles = load_llm_profiles()
    llm_profile_name = st.session_state.get("ollama_profile", settings.ollama_profile)
    llm_profile = llm_profiles.get(llm_profile_name, {})
    ollama_model = str(llm_profile.get("alias", settings.ollama_model))
    index_state_mtime = PREBUILT_INDEX_STATE.stat().st_mtime_ns if PREBUILT_INDEX_STATE.exists() else 0
    return cached_lab(
        signatures,
        index_state_mtime,
        strategy or st.session_state.get("chunking_strategy", "recursive"),
        int(st.session_state.get("chunk_size", 500)),
        int(st.session_state.get("chunk_overlap", 100)),
        float(st.session_state.get("semantic_threshold", 0.72)),
        model,
        embedding_device or st.session_state.get("embedding_device", str(settings.runtime.embedding_device)),
        vector_device or st.session_state.get("vector_device", str(settings.runtime.vector_device)),
        st.session_state.get("llm_provider", "dummy"),
        ollama_model,
        fallback_embedding,
        st.session_state.get("fusion_mode", "rrf"),
        int(st.session_state.get("rrf_k", 60)),
        float(st.session_state.get("dense_weight", 0.5)),
        settings.ollama_connect_timeout_seconds,
        settings.ollama_read_timeout_seconds,
        settings.ollama_max_retries,
        int(llm_profile.get("num_predict", settings.ollama_num_predict)),
        float(llm_profile.get("temperature", 0.15)),
        str(llm_profile.get("keep_alive", "5m")),
    )


def get_selected_retriever(lab):
    mode = st.session_state.get("retrieval_mode", "hybrid")
    if mode == "dense":
        return lab.dense
    if mode == "bm25":
        return lab.sparse
    return lab.hybrid


def _sync_context_budget_from_profile() -> None:
    profile = st.session_state.get("context_profile", "balanced")
    budget = int(CONTEXT_PROFILES.get(profile, CONTEXT_PROFILES["balanced"])["budget"])
    st.session_state["context_budget"] = budget


def sidebar_runtime_controls() -> None:
    settings = load_settings()
    st.sidebar.markdown("### Aktív folyamat konfiguráció")
    st.sidebar.caption(
        "Ezek a beállítások közösek a laboroldalak között, és az összes releváns folyamatlépést módosítják."
    )

    with st.sidebar.expander("1. Darabolás (chunking)", expanded=False):
        st.selectbox(
            "Stratégia",
            list(CHUNKING_STRATEGIES),
            key="chunking_strategy",
            format_func=lambda x: CHUNKING_STRATEGIES[x]["name"],
        )
        st.slider("Szövegrész mérete (chunk size)", 100, 2000, step=50, key="chunk_size")
        max_overlap = max(0, min(500, int(st.session_state.get("chunk_size", 500)) - 1))
        if int(st.session_state.get("chunk_overlap", 100)) > max_overlap:
            st.session_state["chunk_overlap"] = max_overlap
        st.slider("Átfedés", 0, max_overlap, step=10, key="chunk_overlap")
        if st.session_state.get("chunking_strategy") == "semantic":
            st.slider("Szemantikus küszöb", 0.0, 1.0, step=0.01, key="semantic_threshold")
        st.caption(CHUNKING_STRATEGIES[st.session_state.get("chunking_strategy", "recursive")]["summary"])

    with st.sidebar.expander("2. Beágyazás és vektorindex", expanded=False):
        st.selectbox(
            "Beágyazási mód",
            list(EMBEDDING_LABELS),
            key="embedding_mode",
            format_func=lambda x: EMBEDDING_LABELS[x],
            help="A magyar orvosi index alapértelmezett modellje a Multilingual E5 Small; így a perzisztens index újrahasznosítható.",
        )
        st.selectbox("Beágyazási eszköz", ["auto", "cpu", "cuda"], key="embedding_device")
        st.selectbox("FAISS / vektorkeresés eszköz", ["cpu", "cuda"], key="vector_device")

    with st.sidebar.expander("3. Visszakeresés", expanded=False):
        st.selectbox(
            "Visszakereső",
            ["dense", "bm25", "hybrid"],
            key="retrieval_mode",
            format_func=lambda x: {
                "dense": "Dense / beágyazás",
                "bm25": "BM25 / lexikális",
                "hybrid": "Hibrid / dense + BM25",
            }[x],
        )
        st.slider("Top-K", 1, 20, key="top_k")
        min_candidate = int(st.session_state.get("top_k", 5))
        if int(st.session_state.get("candidate_count", 20)) < min_candidate:
            st.session_state["candidate_count"] = min_candidate
        st.slider("Jelöltek száma", min_candidate, 60, key="candidate_count")
        if st.session_state.get("retrieval_mode") == "hybrid":
            st.selectbox(
                "Rangfúzió",
                ["rrf", "weighted"],
                key="fusion_mode",
                format_func=lambda x: "Reciprocal Rank Fusion (RRF)" if x == "rrf" else "Súlyozott pontszámfúzió",
            )
            if st.session_state.get("fusion_mode") == "rrf":
                st.slider("RRF k", 1, 120, key="rrf_k")
            else:
                st.slider("Dense súlya", 0.0, 1.0, step=0.05, key="dense_weight")

    with st.sidebar.expander("4. RAG és válaszgenerálás", expanded=False):
        st.selectbox(
            "RAG stratégia",
            list(RAG_STRATEGIES),
            key="rag_strategy",
            format_func=lambda x: RAG_STRATEGIES[x]["name"],
        )
        st.selectbox(
            "Újrarangsoroló",
            ["lexical", "cross-encoder", "none"],
            key="reranker_mode",
            format_func=lambda x: {
                "lexical": "Lexikális / CPU",
                "cross-encoder": "Cross-Encoder / CPU vagy CUDA",
                "none": "Nincs",
            }[x],
        )
        st.selectbox("Újrarangsoroló eszköz", ["auto", "cpu", "cuda"], key="reranker_device")
        st.slider("Kontextus tokenkeret", 256, 8192, step=128, key="context_budget")
        st.selectbox(
            "LLM szolgáltató",
            ["dummy", "ollama"],
            key="llm_provider",
            format_func=lambda x: "Dummy / offline" if x == "dummy" else "Ollama / lokális Qwen",
            help="Dummy módban a teljes pipeline külső LLM nélkül tesztelhető. Ollama esetén a kiválasztott Qwen runtime profil fut.",
        )
        if st.session_state.get("llm_provider") == "ollama":
            llm_profiles = load_llm_profiles()
            selected_profile_name = resolve_profile_name(
                llm_profiles, st.session_state.get("ollama_profile"), fallback="balanced"
            )
            if st.session_state.get("ollama_profile") != selected_profile_name:
                st.session_state["ollama_profile"] = selected_profile_name
            st.selectbox(
                "Ollama / Qwen profil",
                list(llm_profiles),
                key="ollama_profile",
                format_func=lambda x: str(llm_profiles.get(x, {}).get("display_name", x)),
                help="A profil a Streamlit által használt Ollama modellaliast választja. A teljes futtatási környezetet a RUN/INFRASTRUCTURE launcherek kezelik.",
            )
            selected_llm = llm_profiles[
                resolve_profile_name(llm_profiles, st.session_state.get("ollama_profile"), fallback="balanced")
            ]
            st.caption(
                f"Modell: {selected_llm['alias']} · kontextus: {selected_llm['context_length']} · alapmodell: {selected_llm['base_model']}"
            )

    with st.sidebar.expander("5. Kontextus- és promptprofilok", expanded=False):
        st.selectbox(
            "Kontextusprofil",
            list(CONTEXT_PROFILES),
            key="context_profile",
            format_func=lambda x: CONTEXT_PROFILES[x]["name"],
            help="A profil irányt ad a kontextus összeállításának és automatikusan frissíti az ajánlott tokenkeretet.",
            on_change=_sync_context_budget_from_profile,
        )
        active_context_profile = CONTEXT_PROFILES[st.session_state.get("context_profile", "balanced")]
        st.caption(
            f"{active_context_profile['summary']} · Ajánlott tokenkeret: {int(active_context_profile['budget'])} token. "
            "Profilváltáskor a tokenkeret automatikusan frissül."
        )
        st.selectbox(
            "Promptprofil",
            list(PROMPT_PROFILES),
            key="prompt_profile",
            format_func=lambda x: PROMPT_PROFILES[x]["name"],
            help="A válasz stílusát és szerkezetét szabályozza ugyanazon bizonyítékok mellett.",
        )
        st.caption(str(PROMPT_PROFILES[st.session_state.get("prompt_profile", "professional")]["summary"]))


@st.cache_resource(show_spinner="Cross-Encoder újrarangsoroló betöltése...")
def cached_cross_encoder(model_name: str, device: str):
    return CrossEncoderReranker(model_name, device=device)

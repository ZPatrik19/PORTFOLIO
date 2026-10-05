from __future__ import annotations

import importlib.metadata as metadata

import streamlit as st

from rag_engine.platform.config import load_settings
from rag_engine.presets import CHUNKING_STRATEGIES, CONTEXT_PROFILES, PROMPT_PROFILES, RAG_STRATEGIES
from rag_engine.generation.provider_factory import create_llm_provider
from rag_engine.platform.profiles import load_llm_profiles
from ui.components.common import EMBEDDING_LABELS
from ui.components.education import status_cards


@st.cache_data(ttl=5, show_spinner=False)
def _safe_health(provider_name: str, model_name: str) -> dict[str, object]:
    settings = load_settings()
    provider = create_llm_provider(
        provider_name,
        base_url=settings.ollama_base_url,
        model_name=model_name,
    )
    return provider.health()


@st.cache_data(ttl=5, show_spinner=False)
def _runtime_packages() -> dict[str, str | bool | None]:
    result: dict[str, str | bool | None] = {
        "faiss_version": None,
        "faiss_gpu_api": False,
        "cuda_python": None,
        "torch_version": None,
        "torch_cuda": None,
        "torch_cuda_available": False,
        "torch_gpu_name": None,
    }
    try:
        import faiss  # type: ignore

        result["faiss_version"] = getattr(faiss, "__version__", "ismeretlen")
        result["faiss_gpu_api"] = hasattr(faiss, "StandardGpuResources")
    except Exception:
        pass
    try:
        import cuda.bindings  # type: ignore

        result["cuda_python"] = "telepítve"
    except Exception:
        result["cuda_python"] = None
    try:
        import torch  # type: ignore

        result["torch_version"] = torch.__version__
        result["torch_cuda"] = torch.version.cuda
        result["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            result["torch_gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return result


def render_runtime_badges() -> None:
    settings = load_settings()
    provider_name = st.session_state.get("llm_provider", settings.llm_provider)
    llm_profiles = load_llm_profiles()
    profile_name = st.session_state.get("ollama_profile", settings.ollama_profile)
    model_name = str(llm_profiles.get(profile_name, {}).get("alias", settings.ollama_model))
    llm = _safe_health(provider_name, model_name)
    runtime = _runtime_packages()

    requested_embedding = str(st.session_state.get("embedding_device", "auto"))
    actual_device = "cuda" if runtime["torch_cuda_available"] and requested_embedding in {"auto", "cuda"} else "cpu"

    try:
        cuda_python_version = metadata.version("cuda-python")
    except metadata.PackageNotFoundError:
        cuda_python_version = "nincs"

    if provider_name == "dummy":
        llm_value = "Dummy / offline"
        llm_detail = "Nincs külső LLM-hívás."
        llm_state = "off"
    elif not llm.get("ok"):
        llm_value = "Ollama API nem elérhető"
        llm_detail = f"Várt modell: {model_name}"
        llm_state = "warn"
    elif not llm.get("installed"):
        llm_value = "Ollama fut · modell hiányzik"
        llm_detail = f"Hiányzó alias: {model_name}"
        llm_state = "warn"
    else:
        llm_value = "Ollama + Qwen rendben"
        llm_detail = f"Aktív modellalias: {model_name}"
        llm_state = "ok"

    torch_version = str(runtime.get("torch_version") or "nincs")
    torch_cuda = str(runtime.get("torch_cuda") or "CPU build / nincs CUDA runtime")
    faiss_version = str(runtime.get("faiss_version") or "nincs")

    status_cards(
        [
            (
                "NVIDIA cuda-python",
                f"Telepítve · {cuda_python_version}" if runtime["cuda_python"] else "Nincs telepítve",
                "CUDA Driver/Runtime Python binding; önmagában nem teszi a PyTorchot GPU-képessé.",
                "ok" if runtime["cuda_python"] else "warn",
            ),
            (
                "PyTorch CUDA",
                f"CUDA {torch_cuda}" if runtime["torch_cuda"] else "CPU-only PyTorch",
                f"torch {torch_version} · torch.cuda.is_available() = {'igen' if runtime['torch_cuda_available'] else 'nem'}",
                "ok" if runtime["torch_cuda_available"] else "warn",
            ),
            (
                "Beágyazás futtatása",
                f"{requested_embedding} → {actual_device}",
                f"Kért eszköz: {requested_embedding} · tényleges eszköz: {actual_device}",
                "ok" if actual_device == "cuda" else "info",
            ),
            (
                "FAISS vektorbackend",
                f"FAISS {faiss_version} · CPU backend OK" if runtime["faiss_version"] else "FAISS nincs telepítve",
                "Windows alatt a faiss-cpu a stabil baseline. A GPU API hiánya nem hiba; az index CPU-n fut.",
                "ok" if runtime["faiss_version"] else "warn",
            ),
            ("LLM runtime", llm_value, llm_detail, llm_state),
            (
                "GPU",
                str(runtime.get("torch_gpu_name") or "PyTorchból nem látható"),
                "Az Ollama ettől függetlenül használhat GPU-t; ezt az Ollama runtime logja mutatja.",
                "ok" if runtime.get("torch_gpu_name") else "info",
            ),
        ],
        columns=2,
    )

    if provider_name == "ollama" and llm.get("ok") and not llm.get("installed"):
        st.warning(
            f"Az Ollama szerver fut, de a `{model_name}` modellalias nincs telepítve. "
            "Futtasd az `SETUP.bat` fájlt, majd a `RUN.bat`-ot."
        )

    if requested_embedding == "cuda" and not runtime["torch_cuda_available"]:
        st.warning(
            "CUDA embeddinget kértél, de a telepített PyTorch nem lát CUDA-t. "
            "Futtasd: `python scripts/infrastructure_cli.py cuda`. A rendszer addig CPU fallbackkal működik."
        )


def runtime_snapshot() -> dict[str, object]:
    chunk_key = st.session_state.get("chunking_strategy", "recursive")
    rag_key = st.session_state.get("rag_strategy", "hybrid")
    context_profile = st.session_state.get("context_profile", "balanced")
    prompt_profile = st.session_state.get("prompt_profile", "professional")
    return {
        "chunking": CHUNKING_STRATEGIES.get(chunk_key, {}).get("name", chunk_key),
        "chunk_size": st.session_state.get("chunk_size", 500),
        "overlap": st.session_state.get("chunk_overlap", 100),
        "embedding": EMBEDDING_LABELS.get(
            st.session_state.get("embedding_mode", "multilingual"),
            st.session_state.get("embedding_mode", "multilingual"),
        ),
        "embedding_device": st.session_state.get("embedding_device", "auto"),
        "vector_device": st.session_state.get("vector_device", "cpu"),
        "retrieval": st.session_state.get("retrieval_mode", "hybrid"),
        "fusion": st.session_state.get("fusion_mode", "rrf"),
        "top_k": st.session_state.get("top_k", 5),
        "candidates": st.session_state.get("candidate_count", 20),
        "reranker": st.session_state.get("reranker_mode", "lexical"),
        "reranker_device": st.session_state.get("reranker_device", "auto"),
        "rag": RAG_STRATEGIES.get(rag_key, {}).get("name", rag_key),
        "context_budget": st.session_state.get("context_budget", 1800),
        "context_profile": CONTEXT_PROFILES.get(context_profile, {}).get("name", context_profile),
        "prompt_profile": PROMPT_PROFILES.get(prompt_profile, {}).get("name", prompt_profile),
        "llm": (
            "dummy"
            if st.session_state.get("llm_provider", "dummy") == "dummy"
            else str(load_llm_profiles().get(st.session_state.get("ollama_profile", "balanced"), {}).get("alias", "ollama"))
        ),
    }

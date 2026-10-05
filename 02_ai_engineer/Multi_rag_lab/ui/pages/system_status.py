from __future__ import annotations

import importlib.metadata as metadata
import importlib.util
import platform

import streamlit as st

from rag_engine.platform.config import load_settings
from rag_engine.generation.provider_factory import create_llm_provider
from rag_engine.platform.hardware import get_hardware_profile
from ui.components.common import active_paths
from ui.components.education import info_cards, kpi_cards, page_intro
from ui.components.runtime_status import render_runtime_badges, runtime_snapshot
from ui.components.tables import safe_dataframe


def _version(name: str) -> str:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "telepítve"))
    except Exception:
        return "nincs telepítve"


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "nincs telepítve"


def _torch_cuda_status() -> tuple[str, str]:
    try:
        import torch

        runtime = torch.version.cuda or "nincs / CPU build"
        available = "igen" if torch.cuda.is_available() else "nem"
        return runtime, available
    except Exception:
        return "ismeretlen", "nem"


def _faiss_backend_status() -> str:
    try:
        import faiss

        version = str(getattr(faiss, "__version__", "ismeretlen"))
        gpu = hasattr(faiss, "StandardGpuResources")
        return f"FAISS {version} · {'GPU API' if gpu else 'CPU backend OK'}"
    except Exception:
        return "FAISS nincs telepítve"


def render() -> None:
    page_intro(
        "10. Rendszerállapot és runtime diagnosztika",
        "Itt külön látod a Python pipeline hardverét, a CUDA Python bindinget, a PyTorch CUDA runtime-ot, a vektorbackendet és az LLM runtime-ot. Ez szándékosan nincs összemosva: a CUDA Python telepítése önmagában még nem jelenti azt, hogy PyTorch, FAISS vagy Ollama GPU-n fut.",
        eyebrow="PYTHON · CUDA PYTHON · PYTORCH CUDA · FAISS · EMBEDDING · OLLAMA · KORPUSZ",
    )
    render_runtime_badges()
    info_cards(
        [
            (
                "CUDA Python",
                "Az NVIDIA `cuda-python` Python bindingot ad a CUDA Driver/Runtime API-khoz. Nem helyettesíti a CUDA-képes PyTorch buildet.",
            ),
            (
                "PyTorch CUDA",
                "A Sentence Transformers és a Cross-Encoder GPU gyorsításához a `torch.cuda.is_available()` ténylegesen igaz kell legyen.",
            ),
            ("FAISS", "A FAISS CPU a hordozható baseline. A Python csomag és a GPU API külön státuszként jelenik meg."),
            (
                "Ollama",
                "Az Ollama külön process/service. A Python CUDA státuszából nem következik az Ollama tényleges GPU használata.",
            ),
        ],
        columns=4,
    )

    settings = load_settings()
    profile = get_hardware_profile(settings.runtime.execution_device)
    snap = runtime_snapshot()
    st.subheader("Hardverprofil")
    kpi_cards(
        [
            ("Python device", profile.resolved_device.upper(), "A pipeline által feloldott futtatási device."),
            ("CPU", f"{profile.cpu_count} mag", "Elérhető logikai CPU erőforrás."),
            ("Rendszermemória", f"{profile.system_memory_mb / 1024:.1f} GB", "Teljes elérhető RAM."),
            ("GPU", profile.gpu_name or "nem látható", "A Python runtime által detektált GPU."),
        ],
        columns=2,
    )
    with st.expander("Részletes hardware profile", expanded=False):
        st.json(profile.model_dump())

    torch_cuda_runtime, torch_cuda_available = _torch_cuda_status()
    st.subheader("GPU és CUDA szoftverrétegek")
    kpi_cards(
        [
            ("NVIDIA cuda-python", _package_version("cuda-python"), "CUDA Python binding verzió."),
            ("PyTorch CUDA runtime", torch_cuda_runtime, f"torch.cuda.is_available(): {torch_cuda_available}"),
            (
                "FAISS backend",
                _faiss_backend_status(),
                "Windows alatt a CPU FAISS teljesen támogatott baseline; a GPU API nem kötelező.",
            ),
        ],
        columns=3,
    )
    st.caption(
        "A négy érték külön jelentésű. Például a `cuda-python` telepítve lehet úgy is, hogy a PyTorch még CPU build vagy a driver nem kompatibilis."
    )

    st.subheader("Szoftverkörnyezet")
    versions = {
        "Python": platform.python_version(),
        "PyTorch": _version("torch"),
        "CUDA Python package": _package_version("cuda-python"),
        "NumPy": _version("numpy"),
        "FAISS": _version("faiss"),
        "faiss-cpu package": _package_version("faiss-cpu"),
        "Sentence Transformers": _version("sentence_transformers"),
        "Streamlit": _version("streamlit"),
    }
    safe_dataframe(
        [{"Komponens": k, "Verzió / státusz": v} for k, v in versions.items()], width="stretch", hide_index=True
    )

    st.subheader("Aktív workflow konfiguráció")
    st.json(snap)

    llm = create_llm_provider(
        st.session_state.get("llm_provider", "dummy"),
        base_url=settings.ollama_base_url,
        model_name=settings.ollama_model,
    )
    st.subheader("LLM futtatási környezet")
    health = llm.health()
    st.json(health)
    if st.session_state.get("llm_provider") == "ollama" and not health.get("ok"):
        st.warning(
            "Az Ollama jelenleg nem elérhető. A retrieval és a laborok ettől még futtathatók; teljes generáláshoz indítsd el az Ollamát vagy válts Dummy módra."
        )

    st.subheader("Aktív korpusz")
    paths = active_paths()
    total_mb = sum(path.stat().st_size for path in paths if path.exists()) / 1024**2
    kpi_cards(
        [
            ("Aktív fájlok", str(len(paths)), "A jelenlegi RAG korpusz fájljainak száma."),
            ("Korpuszméret", f"{total_mb:.1f} MB", "Az aktív forrásfájlok összmérete."),
            (
                "FAISS Python modul",
                "igen" if importlib.util.find_spec("faiss") else "nem",
                "A vector backend importálhatósága.",
            ),
        ],
        columns=3,
    )
    if paths:
        safe_dataframe(
            [{"Fájl": path.name, "Méret MB": round(path.stat().st_size / 1024**2, 2)} for path in paths],
            width="stretch",
            hide_index=True,
        )
    else:
        st.warning(
            "Nincs aktív dokumentum. A Dokumentumok oldalon tölts le magyar korpuszt vagy tölts fel saját fájlt."
        )

from __future__ import annotations

import importlib.metadata as metadata
import platform

import pandas as pd
import streamlit as st

from rag_engine.platform.config import load_settings
from rag_engine.platform.profiles import (
    load_cuda_profiles,
    load_faiss_profiles,
    load_llm_profiles,
    resolve_profile_name,
)
from rag_engine.platform.runtime import (
    create_llm_profile,
    cuda_status,
    faiss_status,
    ollama_health,
    ollama_profile_environment,
    ollama_process_status,
    pull_llm_profile,
    restart_ollama,
    test_ollama_model,
)
from ui.components.common import ROOT, active_paths, cached_lab, get_lab, queue_widget_state_updates
from ui.components.education import info_cards, kpi_cards, note_box, page_intro, section_intro, status_cards
from ui.components.tables import safe_dataframe


def _bool_label(value: object) -> str:
    return "igen" if bool(value) else "nem"


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "nincs"


def render() -> None:
    page_intro(
        "Infrastructure",
        "Az Ollama/Qwen, a FAISS vektorindex és a CUDA gyorsítás külön infrastruktúra-rétegként kezelhető. Itt a profilokat közvetlenül a Streamlitből tudod kiválasztani, ellenőrizni és a futó RAG sessionre alkalmazni.",
        eyebrow="OLLAMA / QWEN · FAISS · CUDA · RUNTIME PROFILES",
    )

    flash = st.session_state.pop("_runtime_flash", None)
    if flash:
        st.success(str(flash))

    llm_profiles = load_llm_profiles()
    faiss_profiles = load_faiss_profiles()
    cuda_profiles = load_cuda_profiles()
    health = ollama_health()
    fs = faiss_status()
    cs = cuda_status()

    status_cards(
        [
            (
                "Ollama / Qwen",
                "Elérhető" if health.get("ok") else "Nem elérhető",
                f"Ollama {health.get('version') or '—'} · {len(health.get('models', []))} modell",
                "ok" if health.get("ok") else "warn",
            ),
            (
                "FAISS backend",
                f"FAISS {fs.get('version') or '—'} · {str(fs.get('backend', 'cpu')).upper()}"
                if fs.get("import_ok")
                else "Nem elérhető",
                f"GPU API: {'igen' if fs.get('gpu_api') else 'nem'} · látható GPU-k: {fs.get('visible_gpus', 0)}",
                "ok" if fs.get("import_ok") else "warn",
            ),
            (
                "PyTorch CUDA",
                "Aktív" if cs.get("torch_cuda_available") else "CPU fallback",
                f"Futtatási környezet: {cs.get('torch_cuda_runtime') or '—'} · {cs.get('gpu_name') or 'GPU nem látható'}",
                "ok" if cs.get("torch_cuda_available") else "warn",
            ),
            (
                "GPU memória",
                f"{cs.get('vram_gb') or '—'} GB",
                "A lokális Qwen, embedding és Cross-Encoder ugyanazért a VRAM-ért versenyezhet.",
                "info",
            ),
        ],
        columns=2,
    )

    llm_tab, faiss_tab, cuda_tab, env_tab = st.tabs(
        ["LLM · Ollama + Qwen", "Vector DB · FAISS", "CUDA · NVIDIA + PyTorch", "Environment · Corpus"]
    )

    with llm_tab:
        section_intro(
            "Ollama / Qwen runtime",
            "A profilok eltérő modellel, kontextusmérettel és memória-beállításokkal dolgoznak. A Streamlit az alias modellt hívja az Ollama API-n keresztül.",
        )
        current_llm_profile = resolve_profile_name(
            llm_profiles, st.session_state.get("ollama_profile"), fallback="balanced"
        )
        selected = st.selectbox(
            "LLM runtime profil",
            list(llm_profiles),
            index=list(llm_profiles).index(current_llm_profile),
            format_func=lambda name: str(llm_profiles.get(name, {}).get("display_name", name)),
            key="runtime_llm_profile_select",
        )
        profile = llm_profiles[selected]
        env = ollama_profile_environment(selected)
        kpi_cards(
            [
                ("Profil", str(profile["display_name"]), "Aktív Ollama/Qwen runtime konfiguráció."),
                ("Base model", str(profile["base_model"]), "A lokálisan letöltött alapmodell."),
                ("Streamlit alias", str(profile["alias"]), "A RAG alkalmazás ezt az alias modellt hívja."),
                (
                    "Kontextus",
                    f"{profile['context_length']} token",
                    "Ollama modell-context; nem azonos a RAG evidence budgettel.",
                ),
                ("KV cache", str(profile["kv_cache_type"]), "Memóriaigény és generálási sebesség kompromisszuma."),
                (
                    "Párhuzamosság",
                    f"{profile['num_parallel']} kérés · max {profile['max_loaded_models']} modell",
                    "4 GB VRAM mellett az 1×1 stabil baseline.",
                ),
            ],
            columns=3,
        )
        st.caption(str(profile["description"]))

        b1, b2, b3, b4, b5 = st.columns(5)
        if b1.button("Profil alkalmazása", type="primary", width="stretch"):
            cached_lab.clear()
            queue_widget_state_updates(llm_provider="ollama", ollama_profile=selected)
            st.session_state["_runtime_flash"] = f"Aktív Streamlit LLM profil: {profile['display_name']}"
            st.rerun()
        if b2.button("Profil runtime indítása", width="stretch"):
            with st.spinner("Ollama újraindítása a kiválasztott profil runtime-változóival..."):
                result = restart_ollama(selected)
            if result.get("ok"):
                st.success("Ollama API elérhető, a profil runtime beállításai alkalmazva.")
            else:
                st.error(str(result.get("error")))
        if b3.button("Base model letöltése", width="stretch"):
            with st.spinner(f"{profile['base_model']} letöltése. Ez több perc és több GB is lehet..."):
                result = pull_llm_profile(selected)
            if result.get("ok"):
                st.success("Alapmodell letöltve.")
            else:
                st.error(str(result.get("stderr") or result.get("error")))
        if b4.button("Profil alias létrehozása", width="stretch"):
            with st.spinner("Ollama Modelfile profil létrehozása..."):
                result = create_llm_profile(selected)
            if result.get("ok"):
                st.success(f"Model alias kész: {profile['alias']}")
            else:
                st.error(str(result.get("stderr") or result.get("error")))
        if b5.button("LLM smoke test", width="stretch"):
            settings = load_settings()
            with st.spinner("Qwen rövid streaming smoke test · cold start esetén ez hosszabb lehet..."):
                result = test_ollama_model(
                    str(profile["alias"]),
                    base_url=settings.ollama_base_url,
                    connect_timeout=settings.ollama_connect_timeout_seconds,
                    read_timeout=settings.ollama_read_timeout_seconds,
                    keep_alive=str(profile.get("keep_alive", "15m")),
                )
            if result.get("ok"):
                elapsed = float(result.get("elapsed_ms") or 0.0)
                st.success(f"{result.get('response') or 'RAG runtime OK'} · {elapsed:.0f} ms")
            else:
                st.error(str(result.get("error")))
                if result.get("detail"):
                    st.caption(str(result.get("detail")))

        current_health = ollama_health()
        models = current_health.get("models", [])
        safe_dataframe(
            pd.DataFrame(
                [
                    {"Tulajdonság": "API", "Érték": "OK" if current_health.get("ok") else "nem elérhető"},
                    {"Tulajdonság": "Ollama verzió", "Érték": current_health.get("version", "—")},
                    {
                        "Tulajdonság": "Kiválasztott alias telepítve",
                        "Érték": "igen"
                        if any(
                            str(name) == str(profile["alias"]) or str(name).startswith(f"{profile['alias']}:")
                            for name in models
                        )
                        else "nem",
                    },
                    {"Tulajdonság": "OLLAMA_MODELS", "Érték": env["OLLAMA_MODELS"]},
                    {"Tulajdonság": "Kontextus", "Érték": env["OLLAMA_CONTEXT_LENGTH"]},
                    {"Tulajdonság": "KV cache", "Érték": env["OLLAMA_KV_CACHE_TYPE"]},
                    {"Tulajdonság": "Flash Attention", "Érték": env["OLLAMA_FLASH_ATTENTION"]},
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        process = ollama_process_status()
        if process.get("ok") and str(process.get("table", "")).strip():
            with st.expander("Ollama modellmemória / CPU–GPU offload", expanded=False):
                st.code(str(process.get("table", "")), language=None)
                st.caption(
                    "A PROCESSOR oszlopból látható, hogy a modell GPU-n, CPU-n vagy megosztva fut-e. A túlzott CPU offload jelentősen növelheti a latency-t."
                )
        note_box(
            "Egyszerű indítás",
            f"Napi használathoz elég a gyökérben a RUN.bat. Külön infrastruktúra-kezeléshez használd a `python scripts/infrastructure_cli.py` diagnosztikát; az aktív Qwen profil itt: {selected}.",
        )

    with faiss_tab:
        section_intro(
            "FAISS vector database",
            "A FAISS nem külön szerver: a Python processben futó index. Itt profilt választasz, ellenőrzöd a backendet és perzisztens indexet építhetsz az aktív korpuszból.",
        )
        current_faiss_profile = resolve_profile_name(
            faiss_profiles, st.session_state.get("faiss_profile"), fallback="cpu"
        )
        selected = st.selectbox(
            "FAISS profil",
            list(faiss_profiles),
            index=list(faiss_profiles).index(current_faiss_profile),
            format_func=lambda name: str(faiss_profiles.get(name, {}).get("display_name", name)),
            key="runtime_faiss_profile_select",
        )
        profile = faiss_profiles[selected]
        info_cards(
            [
                ("Profil", str(profile["display_name"])),
                ("Kért device", str(profile["vector_device"])),
                (
                    "Fallback",
                    "NumPy exact search engedélyezve" if profile.get("fallback_to_numpy") else "nincs fallback",
                ),
                ("Perzisztens index", str(profile["output_dir"])),
            ],
            columns=4,
        )
        st.caption(str(profile["description"]))
        b1, b2 = st.columns(2)
        if b1.button("FAISS profil alkalmazása", type="primary", width="stretch"):
            cached_lab.clear()
            queue_widget_state_updates(
                faiss_profile=selected,
                vector_device=str(profile["vector_device"]),
            )
            st.session_state["_runtime_flash"] = f"Aktív vector device: {profile['vector_device']}"
            st.rerun()
        if b2.button("Aktív korpusz indexének mentése", width="stretch"):
            try:
                with st.spinner("Embedding + indexépítés + mentés..."):
                    lab = get_lab(vector_device=str(profile["vector_device"]))
                    out = ROOT / str(profile["output_dir"])
                    lab.vector_store.save(out)
                st.success(f"Index elmentve: {out}")
                st.json(
                    {
                        "requested_profile": selected,
                        "backend": getattr(lab.vector_store, "backend_name", "unknown"),
                        "actual_device": getattr(lab.vector_store, "device", "unknown"),
                        "vectors": len(lab.ingestion.chunks),
                        "path": str(out),
                    }
                )
            except Exception as exc:
                st.error(f"Indexépítés sikertelen: {exc}")

        current = faiss_status()
        safe_dataframe(
            pd.DataFrame(
                [
                    {"Komponens": "faiss-cpu package", "Érték": current.get("faiss_cpu_package") or "nincs"},
                    {"Komponens": "faiss-gpu package", "Érték": current.get("faiss_gpu_package") or "nincs"},
                    {"Komponens": "Aktív backend", "Érték": str(current.get("backend", "unavailable")).upper()},
                    {"Komponens": "Látható FAISS GPU", "Érték": current.get("visible_gpus", 0)},
                    {"Komponens": "FAISS import", "Érték": "OK" if current.get("import_ok") else "hiba"},
                    {"Komponens": "FAISS verzió", "Érték": current.get("version") or "—"},
                    {
                        "Komponens": "GPU API",
                        "Érték": "elérhető" if current.get("gpu_api") else "nem elérhető / CPU build",
                    },
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        note_box(
            "FAISS kezelés",
            "A FAISS nem külön szerver. Az `python scripts/infrastructure_cli.py verify` ellenőrzi a Python backendet és a perzisztens medical indexet; újraindexeléshez válaszd az Orvosi index újraépítés opciót.",
        )

    with cuda_tab:
        section_intro(
            "CUDA runtime profilok",
            "A CUDA Python binding, a CUDA-képes PyTorch és a FAISS GPU három külön réteg. Az embedding/reranker GPU-t a PyTorch biztosítja; a FAISS device külön konfiguráció.",
        )
        current_cuda_profile = resolve_profile_name(
            cuda_profiles, st.session_state.get("cuda_profile"), fallback="auto"
        )
        selected = st.selectbox(
            "CUDA profil",
            list(cuda_profiles),
            index=list(cuda_profiles).index(current_cuda_profile),
            format_func=lambda name: str(cuda_profiles.get(name, {}).get("display_name", name)),
            key="runtime_cuda_profile_select",
        )
        profile = cuda_profiles[selected]
        info_cards(
            [
                ("Profil", str(profile["display_name"])),
                ("Beágyazás", str(profile["embedding_device"])),
                ("Újrarangsoroló", str(profile["reranker_device"])),
                ("FAISS", str(profile["vector_device"])),
            ],
            columns=4,
        )
        st.caption(str(profile["description"]))
        if st.button("CUDA profil alkalmazása", type="primary", width="stretch"):
            cached_lab.clear()
            queue_widget_state_updates(
                cuda_profile=selected,
                embedding_device=str(profile["embedding_device"]),
                reranker_device=str(profile["reranker_device"]),
                vector_device=str(profile["vector_device"]),
            )
            st.session_state["_runtime_flash"] = "CUDA runtime profil alkalmazva a Streamlit sessionre."
            st.rerun()

        current = cuda_status()
        safe_dataframe(
            pd.DataFrame(
                [
                    {"Réteg": "nvidia-smi", "Állapot": "elérhető" if current.get("nvidia_smi") else "nincs"},
                    {"Réteg": "cuda-python", "Állapot": current.get("cuda_python") or "nincs"},
                    {"Réteg": "PyTorch", "Állapot": current.get("torch") or "nincs"},
                    {
                        "Réteg": "PyTorch CUDA runtime",
                        "Állapot": current.get("torch_cuda_runtime") or "CPU build / nincs",
                    },
                    {"Réteg": "torch.cuda.is_available()", "Állapot": _bool_label(current.get("torch_cuda_available"))},
                    {"Réteg": "GPU", "Állapot": current.get("gpu_name") or "nem látható"},
                    {"Réteg": "VRAM", "Állapot": f"{current.get('vram_gb')} GB" if current.get("vram_gb") else "—"},
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        if current.get("nvidia_smi_output"):
            st.code(str(current["nvidia_smi_output"]), language=None)
        note_box(
            "CUDA kezelés",
            f"A gyökérben csak a `python scripts/infrastructure_cli.py` diagnosztikát használd. Ott külön CUDA javítás/setup és teljes ellenőrzés is van. Aktív profil: {selected}.",
        )
    with env_tab:
        section_intro(
            "Environment és aktív korpusz",
            "A korábbi külön Rendszerállapot oldal tartalma ide került, hogy az infrastruktúra egyetlen helyen legyen diagnosztizálható.",
        )
        paths = active_paths()
        total_mb = sum(path.stat().st_size for path in paths if path.exists()) / 1024**2
        kpi_cards(
            [
                ("Python", platform.python_version(), "Aktív interpreter verzió."),
                ("PyTorch", _package_version("torch"), f"CUDA runtime: {cs.get('torch_cuda_runtime') or '—'}"),
                ("FAISS", _package_version("faiss-cpu"), "Python vector backend."),
                ("Streamlit", _package_version("streamlit"), "UI runtime."),
                (
                    "Sentence Transformers",
                    _package_version("sentence-transformers"),
                    "Embedding / Cross-Encoder stack.",
                ),
                ("Aktív dokumentum", str(len(paths)), f"Korpuszméret: {total_mb:.1f} MB"),
            ],
            columns=3,
        )
        with st.expander("Szoftverkörnyezet részletesen", expanded=False):
            safe_dataframe(
                pd.DataFrame(
                    [
                        {"Komponens": "Python", "Verzió": platform.python_version()},
                        {"Komponens": "cuda-python", "Verzió": _package_version("cuda-python")},
                        {"Komponens": "torch", "Verzió": _package_version("torch")},
                        {"Komponens": "faiss-cpu", "Verzió": _package_version("faiss-cpu")},
                        {"Komponens": "sentence-transformers", "Verzió": _package_version("sentence-transformers")},
                        {"Komponens": "streamlit", "Verzió": _package_version("streamlit")},
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        if paths:
            with st.expander("Aktív korpusz fájljai", expanded=False):
                safe_dataframe(
                    pd.DataFrame(
                        [{"Fájl": path.name, "Méret MB": round(path.stat().st_size / 1024**2, 3)} for path in paths]
                    ),
                    width="stretch",
                    hide_index=True,
                )
        note_box(
            "Egyszerűsített infrastruktúra",
            "Napi indítás: RUN.bat. Javítás/diagnosztika: `python scripts/infrastructure_cli.py`. A Streamlit System menüjében minden runtime-státusz ezen az egy Infrastructure oldalon található.",
        )

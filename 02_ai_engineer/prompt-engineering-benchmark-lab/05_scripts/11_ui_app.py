from __future__ import annotations

import os
import runpy
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

st.set_page_config(
    page_title="Prompt Engineering Benchmark Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
os.chdir(PROJECT_ROOT)
load_dotenv()

from prompt_benchmark.paths import PATHS
from prompt_benchmark.utils.logging import configure_logging
from prompt_benchmark.config import load_yaml  # noqa: E402
from prompt_benchmark.evaluation.parsing import parse_prediction  # noqa: E402
from prompt_benchmark.llm.factory import PROVIDER_CAPABILITIES, SUPPORTED_PROVIDERS, create_llm_client  # noqa: E402
from prompt_benchmark.prompts import get_strategy  # noqa: E402

configure_logging()
cfg = load_yaml(PATHS.configs / "benchmark.yaml")

DEFAULT_LANGUAGE = os.getenv("PEB_UI_LANGUAGE", "hu").strip().lower()
if DEFAULT_LANGUAGE not in {"hu", "en"}:
    DEFAULT_LANGUAGE = "hu"

DEFAULT_MODELS = {
    "mock": "mock-prompt-sensitive-simulator-v2",
    "ollama": os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
    "groq": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    "gemini": os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    "openrouter": os.getenv("OPENROUTER_MODEL", "openrouter/free"),
    "openai": os.getenv("OPENAI_MODEL", ""),
}
KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
}

if "ui_language" not in st.session_state:
    st.session_state["ui_language"] = DEFAULT_LANGUAGE
if "runtime_provider" not in st.session_state:
    st.session_state["runtime_provider"] = os.getenv("LLM_PROVIDER", "mock") if os.getenv("LLM_PROVIDER", "mock") in SUPPORTED_PROVIDERS else "mock"


def tr(hu: str, en: str) -> str:
    return hu if st.session_state["ui_language"] == "hu" else en


with st.sidebar:
    st.markdown("### 🌐 Language / Nyelv")
    language_label = st.selectbox(
        "Felület nyelve / Interface language",
        options=["Magyar", "English"],
        index=0 if st.session_state["ui_language"] == "hu" else 1,
        key="ui_language_selector",
    )
    selected_language = "hu" if language_label == "Magyar" else "en"
    if selected_language != st.session_state["ui_language"]:
        st.session_state["ui_language"] = selected_language
        st.rerun()

    st.divider()
    st.markdown(f"### ⚙️ {tr('Futási beállítások', 'Runtime settings')}")

    provider = st.selectbox(
        "Provider",
        SUPPORTED_PROVIDERS,
        key="runtime_provider",
        help=tr(
            "Ugyanaz a benchmark pipeline fut Mock, lokális vagy cloud LLM-mel.",
            "The same benchmark pipeline runs with Mock, local or cloud LLMs.",
        ),
    )

    model_state_key = f"runtime_model_{provider}"
    if model_state_key not in st.session_state:
        st.session_state[model_state_key] = DEFAULT_MODELS[provider]
    model = st.text_input(
        tr("Modell", "Model"),
        key=model_state_key,
        placeholder=tr("modellazonosító", "model id"),
    ).strip()

    env_name = KEY_ENV.get(provider)
    if env_name:
        api_state_key = f"runtime_api_key_{provider}"
        api_key = st.text_input(
            env_name,
            type="password",
            key=api_state_key,
            placeholder=tr("API-kulcs beillesztése", "Paste API key"),
            help=tr(
                "A beírt kulcs csak az aktuális Streamlit process memóriájában él; a projekt nem menti fájlba.",
                "The entered key lives only in the current Streamlit process memory; the project does not write it to disk.",
            ),
        )
        if api_key:
            os.environ[env_name] = api_key
            st.caption(f"✅ {tr('Kulcs betöltve az aktuális UI sessionbe', 'Key loaded into the current UI session')}")
        elif os.getenv(env_name):
            st.caption(f"✅ {env_name}: {tr('.env/környezeti változóból elérhető', 'available from .env/environment')}")
        else:
            st.caption(f"ℹ️ {tr('Ehhez a providerhez API-kulcs szükséges.', 'This provider requires an API key.')}")
    elif provider == "ollama":
        st.caption(tr("🔓 Nincs API-kulcs: lokális Ollama szolgáltatás.", "🔓 No API key: local Ollama service."))
    else:
        st.caption(tr("🔓 Nincs API-kulcs: offline promptérzékeny szimuláció.", "🔓 No API key: offline prompt-sensitive simulation."))

    caps = PROVIDER_CAPABILITIES[provider]
    use_temperature = st.checkbox(
        tr("temperature felülírás", "Override temperature"),
        value=(provider != "gemini"),
        disabled=not caps.get("temperature", False),
        key=f"runtime_use_temperature_{provider}",
        help=tr(
            "Gemini 3.x normál benchmarknál érdemes a modell defaultját használni; sampling kísérlethez kapcsold be.",
            "For normal Gemini 3.x benchmarks prefer model defaults; enable for an explicit sampling experiment.",
        ),
    )
    default_temp = 1.0 if provider == "gemini" else 0.0
    temperature = st.slider(
        "temperature",
        0.0,
        2.0,
        default_temp,
        0.1,
        disabled=(not use_temperature) or (not caps.get("temperature", False)),
        key=f"runtime_temperature_{provider}",
    )

    use_top_p = st.checkbox(
        tr("top_p felülírás", "Override top_p"),
        value=False,
        disabled=not caps.get("top_p", False),
        key=f"runtime_use_top_p_{provider}",
    )
    top_p = st.slider(
        "top_p",
        0.05,
        1.0,
        0.90,
        0.05,
        disabled=(not use_top_p) or (not caps.get("top_p", False)),
        key=f"runtime_top_p_{provider}",
    )

    use_top_k = st.checkbox(
        tr("top_k felülírás", "Override top_k"),
        value=False,
        disabled=not caps.get("top_k", False),
        key=f"runtime_use_top_k_{provider}",
    )
    top_k = st.slider(
        "top_k",
        1,
        100,
        40,
        1,
        disabled=(not use_top_k) or (not caps.get("top_k", False)),
        key=f"runtime_top_k_{provider}",
    )

    # Aliases expected by the language views. One central runtime configuration
    # is intentionally shared across both languages.
    temperature_enabled = use_temperature
    top_p_enabled = use_top_p
    top_k_enabled = use_top_k

    def runtime_overrides() -> dict[str, object]:
        values: dict[str, object] = {"model": model}
        if use_temperature:
            values["temperature"] = temperature
        if use_top_p:
            values["top_p"] = top_p
        if use_top_k:
            values["top_k"] = top_k
        return values

    with st.expander(tr("Provider képességek", "Provider capabilities")):
        st.json(caps)

    test_clicked = st.button(
        tr("🔌 Kapcsolat tesztelése", "🔌 Test connection"),
        use_container_width=True,
        key="central_provider_connection_test",
    )
    if test_clicked:
        if env_name and not os.getenv(env_name):
            st.error(tr(f"Add meg a {env_name} értékét fent.", f"Enter {env_name} above."))
        elif not model:
            st.error(tr("Adj meg egy modellazonosítót.", "Enter a model ID."))
        else:
            with st.spinner(tr("Teszt request küldése…", "Sending test request…")):
                try:
                    client = create_llm_client(provider, cfg, overrides=runtime_overrides())
                    payload = get_strategy("p0_zero_shot").build("I was charged twice for the same monthly subscription.")
                    response = client.classify(payload)
                    parsed = parse_prediction(response.raw_output, payload.output_mode)
                    if response.error:
                        st.error(f"{tr('Kapcsolati hiba', 'Connection error')}: {response.error}")
                    else:
                        st.success(tr("Provider kapcsolat működik.", "Provider connection works."))
                        st.caption(
                            f"{response.provider} · {response.model} · "
                            f"{response.input_tokens}+{response.output_tokens} tokens · "
                            f"{response.latency_seconds:.3f}s · "
                            f"{tr('válasz', 'prediction')}={parsed.label or response.raw_output[:40]}"
                        )
                except Exception as exc:
                    st.error(f"{type(exc).__name__}: {exc}")

    if provider == "mock":
        st.info(tr(
            "Mock = determinisztikus, promptérzékeny szimuláció. A token és latency értékek szimulált/becsült adatok.",
            "Mock = deterministic prompt-sensitive simulation. Token and latency values are simulated/estimated.",
        ))

    st.caption(tr(
        "Az API-kulcsot nem kell külön .bat fájlban konfigurálni: itt add meg, majd teszteld a kapcsolatot.",
        "No separate API-key .bat is needed: enter the key here and test the connection.",
    ))
    st.divider()

# Execute the localized content view inside this same Streamlit runtime. Runtime
# settings above remain identical when the language changes.
view_name = "11_ui_view_hu.py" if st.session_state["ui_language"] == "hu" else "11_ui_view_en.py"
view_path = SCRIPT_DIR / view_name
runpy.run_path(
    str(view_path),
    init_globals={**globals(), "__file__": str(view_path)},
    run_name="__prompt_benchmark_ui_view__",
)

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_engine.platform.profiles import get_profile, load_llm_profiles
from rag_engine.platform.model_assets import model_cache_status
from rag_engine.platform.runtime import (
    create_llm_profile,
    cuda_status,
    faiss_status,
    llm_profile_needs_refresh,
    ollama_health,
    ollama_process_status,
    pull_llm_profile,
    start_ollama,
    test_ollama_model,
)

LOG_ROOT = ROOT / "logs" / "infrastructure"
MEDICAL_INDEX_STATE = ROOT / "artifacts" / "indexes" / "hungarian_medical" / "index_state.json"
MEDICAL_MANIFEST = ROOT / "data" / "raw" / "hungarian_medical" / "manifest.json"


def _logger() -> tuple[logging.Logger, Path]:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = LOG_ROOT / f"infrastructure-{stamp}.log"
    logger = logging.getLogger("multi-rag-infrastructure")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger, path


LOGGER, LOG_PATH = _logger()


def banner(title: str) -> None:
    LOGGER.info("=" * 76)
    LOGGER.info(title)
    LOGGER.info("=" * 76)


def run_stream(command: Iterable[str], *, cwd: Path = ROOT) -> int:
    cmd = [str(x) for x in command]
    LOGGER.info("Parancs: %s", subprocess.list2cmdline(cmd))
    child_env = os.environ.copy()
    # Force Python subprocesses and UTF-8 emitting CLIs to use one encoding.
    # This avoids Windows cp1250 crashes/mis-decoding for symbols and Hungarian text.
    child_env["PYTHONUTF8"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env.setdefault("LANG", "hu_HU.UTF-8")
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        LOGGER.error("Nem sikerult elinditani a parancsot: %s", exc)
        return 1
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line:
            LOGGER.info("%s", line)
    return process.wait()


def faiss_smoke_test() -> tuple[bool, str]:
    try:
        import faiss  # type: ignore
        import numpy as np

        vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        index = faiss.IndexFlatIP(2)
        index.add(vectors)
        _, ids = index.search(np.asarray([[1.0, 0.0]], dtype=np.float32), 1)
        if ids.shape != (1, 1) or int(ids[0, 0]) != 0:
            return False, "A FAISS import sikerult, de a smoke search hibas eredmenyt adott."
        version = str(getattr(faiss, "__version__", "ismeretlen"))
        gpu_api = hasattr(faiss, "StandardGpuResources")
        visible_gpus = int(faiss.get_num_gpus()) if gpu_api and hasattr(faiss, "get_num_gpus") else 0
        if gpu_api and visible_gpus > 0:
            resources = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(resources, 0, index)
            _, gpu_ids = gpu_index.search(np.asarray([[1.0, 0.0]], dtype=np.float32), 1)
            if gpu_ids.shape != (1, 1) or int(gpu_ids[0, 0]) != 0:
                return False, "A FAISS GPU API elerheto, de a GPU smoke search hibas eredmenyt adott."
            return True, f"FAISS {version} GPU backend OK; visible_gpus={visible_gpus}"
        return (
            True,
            f"FAISS {version} CPU backend OK; GPU API={'igen, de nincs lathato GPU' if gpu_api else 'nem (Windows CPU buildnel normalis)'}",
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"FAISS import/smoke hiba: {exc}"


def ensure_faiss() -> bool:
    status = faiss_status()
    if status.get("import_ok"):
        ok, message = faiss_smoke_test()
        if ok:
            LOGGER.info("[OK] %s", message)
            return True
        LOGGER.warning("%s", message)

    LOGGER.warning("FAISS nem hasznalhato. Binary wheel telepites indul...")
    rc = run_stream(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--only-binary=:all:",
            "faiss-cpu>=1.14,<2",
        ]
    )
    if rc != 0:
        LOGGER.error("FAISS telepites sikertelen (exit=%s).", rc)
        return False

    ok, message = faiss_smoke_test()
    if ok:
        LOGGER.info("[OK] %s", message)
        return True
    LOGGER.error("%s", message)
    return False


def show_cuda() -> bool:
    status = cuda_status()
    LOGGER.info("cuda-python: %s", status.get("cuda_python") or "nincs")
    LOGGER.info("PyTorch: %s", status.get("torch") or "nincs")
    LOGGER.info("PyTorch CUDA runtime: %s", status.get("torch_cuda_runtime") or "CPU build / nincs")
    LOGGER.info("torch.cuda.is_available(): %s", status.get("torch_cuda_available"))
    LOGGER.info("GPU: %s", status.get("gpu_name") or "PyTorchbol nem lathato")
    if status.get("nvidia_smi_output"):
        LOGGER.info("nvidia-smi: %s", status["nvidia_smi_output"])
    return bool(status.get("torch_cuda_available"))


def _driver_cuda_version() -> tuple[int, int] | None:
    """Return the maximum CUDA runtime advertised by the installed NVIDIA driver."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        completed = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10, check=False)
        match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", completed.stdout or "")
        if match:
            return int(match.group(1)), int(match.group(2))
    except Exception:
        return None
    return None


def _pytorch_cuda_install_spec() -> tuple[str, str] | None:
    """Return a deterministic CUDA PyTorch wheel spec for the detected driver.

    Python 3.14 requires a recent CUDA wheel. We deliberately use the cu126
    channel for drivers advertising CUDA >= 12.6 because it currently publishes
    stable CPython 3.14 Windows/Linux wheels and remains compatible with newer
    NVIDIA drivers. The PyTorch wheel bundles its own CUDA runtime; the value
    reported by ``nvidia-smi`` is used only as the driver compatibility ceiling.
    """
    version = _driver_cuda_version()
    if version is None or version < (12, 6):
        return None
    return (
        "https://download.pytorch.org/whl/cu126",
        "torch==2.14.1+cu126",
    )


def _pytorch_cuda_index_url() -> str | None:
    """Backward-compatible helper used by tests and diagnostics."""
    spec = _pytorch_cuda_install_spec()
    return spec[0] if spec else None


def install_cuda_pytorch() -> bool:
    """Install/repair a CUDA-enabled PyTorch wheel inside the project venv.

    ``pip install --upgrade torch`` is not sufficient when a newer CPU-only
    PyPI build is already installed: pip may keep it because its public version
    is newer than a CUDA wheel on another index. We therefore install an explicit
    CUDA build with ``--force-reinstall`` and validate the result afterwards.
    This never installs or changes the NVIDIA display driver.
    """
    spec = _pytorch_cuda_install_spec()
    if spec is None:
        version = _driver_cuda_version()
        if version is None:
            LOGGER.warning("NVIDIA driver / nvidia-smi nem lathato. CUDA PyTorch nem telepitheto automatikusan.")
        else:
            LOGGER.error(
                "A driver altal jelzett CUDA %s.%s tul regi a Python 3.14-es CUDA wheelhez. "
                "Frissitsd az NVIDIA drivert legalabb CUDA 12.6 kompatibilis verziora.",
                version[0],
                version[1],
            )
        return False

    index_url, torch_spec = spec
    LOGGER.info("CUDA-s PyTorch telepitese/javitasa: %s @ %s", torch_spec, index_url)
    rc = run_stream(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-cache-dir",
            torch_spec,
            "--index-url",
            index_url,
        ]
    )
    if rc != 0:
        LOGGER.error("CUDA PyTorch telepites sikertelen (exit=%s).", rc)
        return False

    if show_cuda():
        LOGGER.info("[OK] CUDA-s PyTorch aktiv.")
        return True

    LOGGER.error(
        "A CUDA PyTorch wheel telepult, de torch.cuda.is_available() tovabbra is False. "
        "Ellenorizd az NVIDIA drivert es indits uj konzolt/gepet, ha a driver frissult."
    )
    return False


def setup_cuda(*, install_if_possible: bool = False) -> bool:
    """Validate CUDA and optionally repair the PyTorch CUDA wheel in the venv."""
    if show_cuda():
        LOGGER.info("[OK] PyTorch CUDA aktiv; kulon CUDA setup nem szukseges.")
        return True

    status = cuda_status()
    if not status.get("nvidia_smi"):
        LOGGER.warning("NVIDIA driver nem lathato; a projekt CPU fallbackkal futtathato.")
        return False

    LOGGER.warning("NVIDIA GPU/driver lathato, de a jelenlegi PyTorch build nem eri el a CUDA-t.")
    if install_if_possible:
        return install_cuda_pytorch()

    LOGGER.info(
        "Automatikus javitashoz hasznald az INFRASTRUCTURE menut vagy: "
        "python scripts/infrastructure_cli.py cuda --install-cuda yes"
    )
    return False


def install_ollama() -> bool:
    """Install Ollama with the vendor-provided installer when it is missing."""
    if shutil.which("ollama"):
        return True
    if sys.platform.startswith("win"):
        LOGGER.info("Ollama telepitese a hivatalos Windows PowerShell installerrel...")
        rc = run_stream(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                "irm https://ollama.com/install.ps1 | iex",
            ]
        )
    else:
        if shutil.which("curl") is None:
            LOGGER.error("curl nem talalhato; Ollama automatikus telepites nem indithato.")
            return False
        LOGGER.info("Ollama telepitese a hivatalos Linux installerrel...")
        rc = run_stream(["bash", "-lc", "curl -fsSL https://ollama.com/install.sh | sh"])
    if rc != 0:
        return False
    # Windows installer paths are not always visible to the already-running Python process.
    if sys.platform.startswith("win"):
        local = Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "Ollama"
        if local.exists():
            os.environ["PATH"] = str(local) + os.pathsep + os.environ.get("PATH", "")
    return shutil.which("ollama") is not None


def _model_installed(alias: str, models: list[str]) -> bool:
    return any(name == alias or name.startswith(f"{alias}:") for name in models)


def ensure_llm(profile_name: str) -> bool:
    profiles = load_llm_profiles()
    profile = get_profile(profiles, profile_name, kind="LLM")
    alias = str(profile["alias"])
    base_model = str(profile["base_model"])

    started = start_ollama(profile_name)
    if not started.get("ok"):
        LOGGER.error("Ollama nem indithato: %s", started.get("error", started))
        return False

    health = ollama_health()
    models = [str(x) for x in health.get("models", [])]
    alias_exists = _model_installed(alias, models)
    if not alias_exists:
        LOGGER.info("A %s alias meg nincs telepitve. Base model letoltes: %s", alias, base_model)
        pulled = pull_llm_profile(profile_name)
        if not pulled.get("ok"):
            LOGGER.error("Ollama pull sikertelen: %s", pulled.get("stderr") or pulled.get("error") or pulled)
            return False
        LOGGER.info("Qwen base model letoltve. Alias letrehozasa: %s", alias)
        created = create_llm_profile(profile_name)
        if not created.get("ok"):
            LOGGER.error("Ollama alias create sikertelen: %s", created.get("stderr") or created.get("error") or created)
            return False
    elif llm_profile_needs_refresh(profile_name):
        LOGGER.info("A %s modellprofil modositva lett; alias automatikus frissitese...", alias)
        created = create_llm_profile(profile_name)
        if not created.get("ok"):
            LOGGER.error(
                "Ollama alias frissites sikertelen: %s", created.get("stderr") or created.get("error") or created
            )
            return False
        LOGGER.info("[OK] Ollama alias frissitve az aktualis magyar grounded profilra: %s", alias)

    health = ollama_health()
    models = [str(x) for x in health.get("models", [])]
    if not _model_installed(alias, models):
        LOGGER.error("Az Ollama fut, de a modellalias tovabbra sem lathato: %s", alias)
        return False

    LOGGER.info("[OK] Ollama %s; modellalias: %s", health.get("version", "?"), alias)
    return True


def llm_smoke(profile_name: str) -> bool:
    profile = get_profile(load_llm_profiles(), profile_name, kind="LLM")
    alias = str(profile["alias"])
    result = test_ollama_model(alias)
    if not result.get("ok"):
        LOGGER.error("LLM smoke test sikertelen: %s", result.get("error", result))
        return False
    LOGGER.info("[OK] Qwen smoke valasz: %s", result.get("response", ""))
    process_status = ollama_process_status()
    if process_status.get("ok") and process_status.get("table"):
        LOGGER.info("Ollama processzor/offload állapot:\n%s", process_status["table"])
    return True


def ensure_runtime_models(*, verify: bool = True) -> bool:
    banner("HUGGING FACE RUNTIME MODEL CACHE")
    command = [sys.executable, str(ROOT / "scripts" / "prepare_runtime_assets.py"), "--retries", "5"]
    if verify:
        command.append("--verify")
    rc = run_stream(command)
    if rc != 0:
        LOGGER.error("A Hugging Face runtime model cache előkészítése sikertelen.")
        return False
    status = model_cache_status()
    LOGGER.info("[OK] HF model cache: %s/%s modell kész.", status.get("ready"), status.get("total"))
    return bool(status.get("complete"))


def ensure_medical_index(*, force: bool = False) -> bool:
    banner("MEDICAL CORPUS + FAISS INDEX")
    command = [sys.executable, str(ROOT / "scripts" / "prepare_medical_corpus.py")]
    if force:
        command.append("--force-index")
    rc = run_stream(command)
    if rc == 0 and MEDICAL_INDEX_STATE.exists():
        LOGGER.info("[OK] Medical corpus és perzisztens index kész.")
        return True
    LOGGER.error("A medical corpus/index előállítása nem fejeződött be sikeresen.")
    return False


def show_faiss_index() -> None:
    if MEDICAL_INDEX_STATE.exists():
        LOGGER.info("[OK] Medical vector index state: %s", MEDICAL_INDEX_STATE)
        try:
            payload = json.loads(MEDICAL_INDEX_STATE.read_text(encoding="utf-8"))
            for key in ("documents", "chunks", "embedding_model", "vector_device", "fingerprint"):
                if key in payload:
                    LOGGER.info("    %s = %s", key, payload[key])
        except Exception as exc:
            LOGGER.warning("Az index_state.json nem olvashato teljesen: %s", exc)
    elif MEDICAL_MANIFEST.exists():
        LOGGER.warning(
            "Medical corpus mar letezik, de nincs perzisztens index. Használd: python scripts/infrastructure_cli.py reindex"
        )
    else:
        LOGGER.warning(
            "Medical corpus/index meg nincs letoltve. Ez nem FAISS telepitesi hiba. A Dokumentumok oldalon vagy a python scripts/infrastructure_cli.py reindex paranccsal keszitheto el."
        )


def reindex() -> bool:
    return ensure_medical_index(force=True)


def setup_all(profile: str, *, install_cuda: bool = False, install_ollama_runtime: bool = False) -> int:
    banner("MULTI-RAG INFRASTRUCTURE SETUP")
    LOGGER.info("Log: %s", LOG_PATH)
    LOGGER.info("[1/5] CUDA / PyTorch")
    cuda_ok = setup_cuda(install_if_possible=install_cuda)
    if not cuda_ok:
        LOGGER.warning("CUDA nem teljesen aktiv. A projekt CPU fallbackkal futtathato.")

    LOGGER.info("[2/5] FAISS vector backend")
    if not ensure_faiss():
        return 2

    LOGGER.info("[3/5] Hugging Face embedding + reranker model cache")
    if not ensure_runtime_models(verify=True):
        return 4

    LOGGER.info("[4/5] Ollama / Qwen (%s)", profile)
    if shutil.which("ollama") is None:
        if install_ollama_runtime:
            if not install_ollama():
                LOGGER.error("Ollama automatikus telepitese sikertelen.")
                return 3
        else:
            LOGGER.warning("Ollama telepites kihagyva. Dummy providerrel a projekt tovabbra is futtathato.")
    if shutil.which("ollama") is not None:
        if not ensure_llm(profile):
            return 3

    LOGGER.info("[5/5] Magyar orvosi korpusz + perzisztens FAISS index")
    if not ensure_medical_index(force=False):
        return 5

    show_faiss_index()
    LOGGER.info(
        "[OK] Infrastruktur setup befejezve; a runtime modellek és az alapértelmezett index elő vannak készítve."
    )
    return 0


def run_all(profile: str) -> int:
    banner("MULTI-RAG INFRASTRUCTURE RUNTIME")
    LOGGER.info("Log: %s", LOG_PATH)
    show_cuda()
    if not ensure_faiss():
        LOGGER.error("A FAISS backend nem hasznalhato, a RAG pipeline nem indul biztonsagosan.")
        return 2
    cache = model_cache_status()
    if not cache.get("complete"):
        LOGGER.warning("A HF runtime modell-cache nem teljes; egyszeri javító letöltés indul.")
        if not ensure_runtime_models(verify=True):
            return 4
    if not MEDICAL_INDEX_STATE.exists():
        LOGGER.warning("A perzisztens orvosi index hiányzik; egyszeri korpusz/index build indul.")
        if not ensure_medical_index(force=False):
            return 5
    show_faiss_index()
    if not ensure_llm(profile):
        return 3
    LOGGER.info("[OK] Infrastruktur runtime kesz; model-cache és perzisztens index rendelkezésre áll.")
    return 0


def verify_all(profile: str) -> int:
    banner("MULTI-RAG INFRASTRUCTURE STATUS")
    cuda_ok = show_cuda()
    faiss_ok = ensure_faiss()
    show_faiss_index()
    llm_ok = ensure_llm(profile)
    if llm_ok:
        llm_smoke(profile)
    LOGGER.info(
        "Osszesites: CUDA=%s | FAISS=%s | OLLAMA/QWEN=%s",
        "OK" if cuda_ok else "CPU fallback",
        "OK" if faiss_ok else "HIBA",
        "OK" if llm_ok else "HIBA",
    )
    return 0 if faiss_ok and llm_ok else 1


def _pause() -> None:
    try:
        input("\nNyomj ENTER-t a folytatashoz...")
    except EOFError:
        pass


def interactive_menu(default_profile: str) -> int:
    while True:
        print("\n" + "=" * 76)
        print(" MULTI-RAG INFRASTRUCTURE")
        print("=" * 76)
        print(" [1] Setup / javitas      CUDA + FAISS + Ollama/Qwen")
        print(" [2] Runtime ellenorzes   napi inditas elotti check")
        print(" [3] Teljes diagnosztika  minden statusz + LLM smoke")
        print(" [4] HF embedding/reranker modellek letöltése + ellenőrzése")
        print(" [5] Orvosi korpusz/index letöltés + újraindexelés")
        print(" [6] Csak CUDA javítás")
        print(" [7] Csak FAISS javítás / smoke test")
        print(" [8] Csak Ollama/Qwen ellenőrzés")
        print(" [0] Kilepes")
        print(f"\n Aktualis LLM profil: {default_profile}")
        print(f" Log: {LOG_PATH}")
        choice = input("\nValassz: ").strip()
        try:
            if choice == "1":
                rc = setup_all(default_profile, install_cuda=True, install_ollama_runtime=True)
            elif choice == "2":
                rc = run_all(default_profile)
            elif choice == "3":
                rc = verify_all(default_profile)
            elif choice == "4":
                rc = 0 if ensure_runtime_models(verify=True) else 1
            elif choice == "5":
                rc = 0 if reindex() else 1
            elif choice == "6":
                rc = 0 if setup_cuda(install_if_possible=True) else 1
            elif choice == "7":
                rc = 0 if ensure_faiss() else 1
            elif choice == "8":
                rc = 0 if ensure_llm(default_profile) else 1
                if rc == 0:
                    llm_smoke(default_profile)
            elif choice == "0":
                LOGGER.info("Kilepes.")
                return 0
            else:
                print("Ismeretlen valasztas.")
                continue
        except Exception:
            LOGGER.exception("Varatlan infrastructure hiba")
            rc = 99
        print(f"\nMuvelet vege. Exit code: {rc}")
        print(f"Reszletes log: {LOG_PATH}")
        _pause()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-RAG infrastructure manager")
    parser.add_argument(
        "command",
        nargs="?",
        default="menu",
        choices=["menu", "setup", "run", "verify", "models", "reindex", "cuda", "faiss", "llm"],
    )
    parser.add_argument("profile", nargs="?", default="balanced")
    parser.add_argument("--install-cuda", choices=["auto", "yes", "no"], default="auto")
    parser.add_argument("--install-ollama", choices=["auto", "yes", "no"], default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    LOGGER.info("Infrastructure CLI indul. command=%s profile=%s", args.command, args.profile)
    LOGGER.info("Log file: %s", LOG_PATH)
    try:
        if args.command == "menu":
            return interactive_menu(args.profile)
        if args.command == "setup":
            cuda_install = args.install_cuda == "yes" or (
                args.install_cuda == "auto" and cuda_status().get("nvidia_smi")
            )
            ollama_install = args.install_ollama in {"auto", "yes"}
            return setup_all(
                args.profile,
                install_cuda=bool(cuda_install),
                install_ollama_runtime=bool(ollama_install),
            )
        if args.command == "run":
            return run_all(args.profile)
        if args.command == "verify":
            return verify_all(args.profile)
        if args.command == "models":
            return 0 if ensure_runtime_models(verify=True) else 1
        if args.command == "reindex":
            return 0 if reindex() else 1
        if args.command == "cuda":
            cuda_install = args.install_cuda in {"auto", "yes"}
            return 0 if setup_cuda(install_if_possible=cuda_install) else 1
        if args.command == "faiss":
            return 0 if ensure_faiss() else 1
        if args.command == "llm":
            return 0 if ensure_llm(args.profile) else 1
        return 2
    except KeyboardInterrupt:
        LOGGER.warning("Megszakitva a felhasznalo altal.")
        return 130
    except Exception:
        LOGGER.exception("Nem kezelt infrastructure hiba")
        return 99


if __name__ == "__main__":
    raise SystemExit(main())

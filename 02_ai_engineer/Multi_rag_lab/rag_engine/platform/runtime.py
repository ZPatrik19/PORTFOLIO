from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time
from typing import Any

import requests

from rag_engine.platform.profiles import CONFIG_ROOT, ROOT, get_profile, load_llm_profiles


LOG_DIR = ROOT / "logs"

PROFILE_STATE_PATH = ROOT / ".cache" / "ollama_profile_state.json"


def _profile_modelfile(profile_name: str) -> Path:
    return CONFIG_ROOT / "ollama" / f"Modelfile.{profile_name}"


def _profile_fingerprint(profile_name: str) -> str:
    modelfile = _profile_modelfile(profile_name)
    if not modelfile.exists():
        return ""
    return hashlib.sha256(modelfile.read_bytes()).hexdigest()


def _load_profile_state() -> dict[str, str]:
    if not PROFILE_STATE_PATH.exists():
        return {}
    try:
        payload = json.loads(PROFILE_STATE_PATH.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in payload.items()} if isinstance(payload, dict) else {}
    except OSError, json.JSONDecodeError:
        return {}


def _write_profile_state(profile_name: str) -> None:
    state = _load_profile_state()
    state[profile_name] = _profile_fingerprint(profile_name)
    PROFILE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def llm_profile_needs_refresh(profile_name: str) -> bool:
    current = _profile_fingerprint(profile_name)
    if not current:
        return False
    return _load_profile_state().get(profile_name) != current


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def ollama_executable() -> str | None:
    return shutil.which("ollama")


def ollama_health(base_url: str = "http://127.0.0.1:11434") -> dict[str, Any]:
    try:
        version = requests.get(f"{base_url.rstrip('/')}/api/version", timeout=2.5)
        version.raise_for_status()
        tags = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=4)
        tags.raise_for_status()
        models = [str(item.get("name", "")) for item in tags.json().get("models", [])]
        return {
            "ok": True,
            "version": version.json().get("version", "ismeretlen"),
            "models": models,
            "base_url": base_url,
        }
    except requests.RequestException as exc:
        return {"ok": False, "base_url": base_url, "error": str(exc), "models": []}


def ollama_profile_environment(profile_name: str) -> dict[str, str]:
    profile = get_profile(load_llm_profiles(), profile_name, kind="LLM")
    models_dir = Path(os.getenv("OLLAMA_MODELS", str(ROOT / ".cache" / "ollama_models")))
    models_dir.mkdir(parents=True, exist_ok=True)
    return {
        "OLLAMA_MODELS": str(models_dir),
        "OLLAMA_CONTEXT_LENGTH": str(profile["context_length"]),
        "OLLAMA_NUM_PARALLEL": str(profile["num_parallel"]),
        "OLLAMA_MAX_LOADED_MODELS": str(profile["max_loaded_models"]),
        "OLLAMA_FLASH_ATTENTION": "1" if profile.get("flash_attention", True) else "0",
        "OLLAMA_KV_CACHE_TYPE": str(profile.get("kv_cache_type", "q8_0")),
        "OLLAMA_KEEP_ALIVE": str(profile.get("keep_alive", "5m")),
        "OLLAMA_HOST": "127.0.0.1:11434",
    }


def start_ollama(profile_name: str) -> dict[str, Any]:
    executable = ollama_executable()
    if not executable:
        return {
            "ok": False,
            "error": "Az ollama parancs nem található. Futtasd a SETUP.bat fájlt, majd ellenőrzéshez használd: python scripts/infrastructure_cli.py verify.",
        }
    health = ollama_health()
    if health.get("ok"):
        return {"ok": True, "already_running": True, **health}

    env = os.environ.copy()
    env.update(ollama_profile_environment(profile_name))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"ollama-{profile_name}.log"
    log_handle = log_path.open("a", encoding="utf-8")
    kwargs: dict[str, Any] = {
        "cwd": str(ROOT),
        "env": env,
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
    }
    if platform.system() == "Windows":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen([executable, "serve"], **kwargs)
    for _ in range(20):
        time.sleep(0.5)
        health = ollama_health()
        if health.get("ok"):
            return {"ok": True, "started": True, "log": str(log_path), **health}
    return {"ok": False, "error": "Ollama elindult, de az API nem lett elérhető időben.", "log": str(log_path)}


def restart_ollama(profile_name: str) -> dict[str, Any]:
    """Restart local Ollama so profile-level environment variables take effect."""
    if platform.system() == "Windows":
        subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], capture_output=True, text=True, check=False)
        time.sleep(0.8)
    else:
        subprocess.run(["pkill", "-f", "ollama serve"], capture_output=True, text=True, check=False)
        time.sleep(0.8)
    return start_ollama(profile_name)


ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _decode_subprocess_output(value: bytes | str | None) -> str:
    """Decode CLI output safely across Windows code pages and UTF-8 tools.

    Ollama is a UTF-8 emitting Go CLI, while Windows Python may otherwise use
    the active ANSI code page (for example cp1250) when text=True is enabled.
    Capture bytes and decode explicitly so progress characters cannot crash the
    infrastructure manager.
    """
    if value is None:
        return ""
    if isinstance(value, bytes):
        decoded = value.decode("utf-8", errors="replace")
    else:
        decoded = value
    decoded = ANSI_ESCAPE_RE.sub("", decoded)
    # Ollama progress output frequently redraws the same line with carriage returns.
    return decoded.replace("\r", "\n").strip()


def _run_ollama(
    args: list[str],
    *,
    timeout: float = 3600,
    stream_output: bool = False,
) -> dict[str, Any]:
    executable = ollama_executable()
    if not executable:
        return {"ok": False, "error": "Az ollama parancs nem található."}
    try:
        if stream_output:
            completed = subprocess.run(
                [executable, *args],
                cwd=ROOT,
                timeout=timeout,
                check=False,
            )
            return {
                "ok": completed.returncode == 0,
                "returncode": completed.returncode,
                "stdout": "",
                "stderr": "",
            }

        completed = subprocess.run(
            [executable, *args],
            cwd=ROOT,
            capture_output=True,
            text=False,
            timeout=timeout,
            check=False,
        )
        stdout = _decode_subprocess_output(completed.stdout)
        stderr = _decode_subprocess_output(completed.stderr)
        result: dict[str, Any] = {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
        if completed.returncode != 0 and not stderr and stdout:
            result["error"] = stdout
        return result
    except subprocess.TimeoutExpired as exc:
        stdout = _decode_subprocess_output(exc.stdout)
        stderr = _decode_subprocess_output(exc.stderr)
        return {
            "ok": False,
            "error": f"Időtúllépés az Ollama parancsnál: {' '.join(args)}",
            "stdout": stdout,
            "stderr": stderr,
        }
    except OSError as exc:
        return {"ok": False, "error": f"Az Ollama parancs nem futtatható: {exc}"}


def pull_llm_profile(profile_name: str) -> dict[str, Any]:
    profile = get_profile(load_llm_profiles(), profile_name, kind="LLM")
    if not ollama_health().get("ok"):
        started = start_ollama(profile_name)
        if not started.get("ok"):
            return started
    return _run_ollama(["pull", str(profile["base_model"])], stream_output=True)


def create_llm_profile(profile_name: str) -> dict[str, Any]:
    profile = get_profile(load_llm_profiles(), profile_name, kind="LLM")
    if not ollama_health().get("ok"):
        started = start_ollama(profile_name)
        if not started.get("ok"):
            return started
    modelfile = _profile_modelfile(profile_name)
    if not modelfile.exists():
        return {"ok": False, "error": f"Modelfile nem található: {modelfile}"}
    result = _run_ollama(["create", str(profile["alias"]), "-f", str(modelfile)])
    if result.get("ok"):
        _write_profile_state(profile_name)
    return result


def ollama_process_status() -> dict[str, Any]:
    """Return the raw `ollama ps` table so GPU/CPU offload is visible in diagnostics."""
    result = _run_ollama(["ps"])
    if not result.get("ok"):
        return result
    return {"ok": True, "table": str(result.get("stdout", "")).strip()}


def test_ollama_model(
    model: str,
    base_url: str = "http://127.0.0.1:11434",
    *,
    connect_timeout: float = 10.0,
    read_timeout: float = 600.0,
    keep_alive: str = "15m",
) -> dict[str, Any]:
    """Run a short streaming smoke test that also tolerates cold model loading.

    A non-streaming 120 second request was too brittle on local hardware: the
    first request can spend most of its time loading the model into RAM/VRAM.
    Streaming plus the same long read timeout used by the real provider makes
    this diagnostic representative without generating a long answer.
    """
    started = time.perf_counter()
    pieces: list[str] = []
    final_event: dict[str, Any] = {}
    try:
        with requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={
                "model": model,
                "prompt": "Válaszolj pontosan ennyit: RAG runtime OK",
                "stream": True,
                "think": False,
                "keep_alive": keep_alive,
                "options": {"num_predict": 16, "temperature": 0.0},
            },
            stream=True,
            timeout=(max(1.0, float(connect_timeout)), max(30.0, float(read_timeout))),
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=False):
                if not raw_line:
                    continue
                try:
                    event = json.loads(raw_line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                if event.get("error"):
                    return {"ok": False, "error": str(event["error"])}
                piece = str(event.get("response", ""))
                if piece:
                    pieces.append(piece)
                if event.get("done"):
                    final_event = event

        return {
            "ok": True,
            "response": "".join(pieces).strip(),
            "eval_count": final_event.get("eval_count"),
            "eval_duration": final_event.get("eval_duration"),
            "load_duration": final_event.get("load_duration"),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
        }
    except requests.Timeout as exc:
        return {
            "ok": False,
            "error": (
                f"Ollama smoke test időtúllépés {read_timeout:.0f}s után. "
                "A modell cold startja vagy CPU fallbackje túl lassú lehet. "
                "Ellenőrizd az `ollama ps` kimenetet, majd próbáld a Low memory profilt."
            ),
            "detail": str(exc),
        }
    except requests.RequestException as exc:
        return {"ok": False, "error": str(exc)}


def cuda_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "cuda_python": package_version("cuda-python"),
        "torch": package_version("torch"),
        "torch_cuda_runtime": None,
        "torch_cuda_available": False,
        "gpu_name": None,
        "vram_gb": None,
        "nvidia_smi": shutil.which("nvidia-smi") is not None,
    }
    try:
        import torch

        status["torch_cuda_runtime"] = torch.version.cuda
        status["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            status["gpu_name"] = torch.cuda.get_device_name(0)
            status["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    except Exception as exc:
        status["torch_error"] = str(exc)
    if status["nvidia_smi"]:
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            status["nvidia_smi_output"] = completed.stdout.strip()
        except Exception as exc:
            status["nvidia_smi_error"] = str(exc)
    return status


def faiss_status() -> dict[str, Any]:
    result: dict[str, Any] = {
        "faiss_cpu_package": package_version("faiss-cpu"),
        "faiss_gpu_package": package_version("faiss-gpu"),
        "import_ok": False,
        "version": None,
        "gpu_api": False,
        "visible_gpus": 0,
        "backend": "unavailable",
    }
    try:
        import faiss

        result["import_ok"] = True
        result["version"] = getattr(faiss, "__version__", "ismeretlen")
        result["gpu_api"] = hasattr(faiss, "StandardGpuResources")
        if result["gpu_api"] and hasattr(faiss, "get_num_gpus"):
            result["visible_gpus"] = int(faiss.get_num_gpus())
        result["backend"] = "gpu" if result["gpu_api"] and result["visible_gpus"] > 0 else "cpu"
    except Exception as exc:
        result["error"] = str(exc)
    return result


def dump_runtime_snapshot(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

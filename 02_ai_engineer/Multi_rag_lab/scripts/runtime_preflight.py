from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_engine.platform.config import load_settings
from rag_engine.platform.runtime import cuda_status, faiss_status, ollama_health


def _run_ollama_preflight(profile: str) -> int:
    command = [sys.executable, str(ROOT / "scripts" / "infrastructure_cli.py"), "llm", profile]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-RAG runtime preflight")
    parser.add_argument(
        "--setup-check",
        action="store_true",
        help="Setup-vegi diagnosztika: az opcionális Ollama hiány nem teszi sikertelenné a setupot.",
    )
    args = parser.parse_args()

    settings = load_settings()
    demo = ROOT / "data" / "demo" / "magyar_rag_demo.md"
    if not demo.exists():
        print(f"[ERROR] Demo dokumentum hianyzik: {demo}")
        return 2

    faiss = faiss_status()
    if faiss.get("import_ok"):
        print(f"[OK] FAISS {faiss.get('version')} · backend={faiss.get('backend')}")
    else:
        print("[WARN] FAISS nem importalhato; a projekt NumPy exact-search fallbackot fog hasznalni.")

    cuda = cuda_status()
    if cuda.get("torch_cuda_available"):
        print(f"[OK] PyTorch CUDA · GPU={cuda.get('gpu_name')} · CUDA runtime={cuda.get('torch_cuda_runtime')}")
    else:
        print("[INFO] PyTorch CUDA nem aktiv; CPU futas hasznalhato.")

    if settings.llm_provider.lower() == "dummy":
        print("[OK] LLM_PROVIDER=dummy · kulso/lokalis LLM runtime nem szukseges.")
        return 0

    if settings.llm_provider.lower() != "ollama":
        print(f"[ERROR] Nem tamogatott LLM provider: {settings.llm_provider}")
        return 3

    health = ollama_health(settings.ollama_base_url)
    if health.get("ok"):
        print(f"[OK] Ollama API elerheto · version={health.get('version')}")
    else:
        print("[INFO] Ollama API meg nem elerheto; inditas/profil-ellenorzes kovetkezik.")

    rc = _run_ollama_preflight(settings.ollama_profile)
    if rc == 0:
        print(f"[OK] Ollama profil kesz: {settings.ollama_profile}")
        return 0

    print(
        "[WARN] Ollama/Qwen runtime nincs teljesen keszen. "
        "Telepitsd/inditsd az Ollamat, vagy allitsd a .env-ben LLM_PROVIDER=dummy ertekre."
    )
    return 0 if args.setup_check else 4


if __name__ == "__main__":
    raise SystemExit(main())

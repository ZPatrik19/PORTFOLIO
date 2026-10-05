from __future__ import annotations

import importlib.metadata as metadata
import shutil
import subprocess
import sys


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "nincs telepítve"


def _nvidia_smi() -> str:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return "nem található"
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout.strip()
        return output or "elérhető, de nem adott vissza GPU-információt"
    except Exception as exc:  # pragma: no cover - diagnostic script
        return f"hiba: {exc}"


def main() -> int:
    print("=" * 72)
    print("Multi-RAG Engineering Lab - gyorsítási környezet ellenőrzése")
    print("=" * 72)
    print(f"Python                  : {sys.version.split()[0]}")
    print(f"faiss-cpu package       : {_version('faiss-cpu')}")
    print(f"cuda-python package     : {_version('cuda-python')}")
    print(f"torch package           : {_version('torch')}")
    print(f"NVIDIA / nvidia-smi     : {_nvidia_smi()}")

    print("\n[FAISS]")
    try:
        import faiss

        print(f"Import                  : OK")
        print(f"FAISS verzió            : {getattr(faiss, '__version__', 'ismeretlen')}")
        print(f"GPU API                 : {'igen' if hasattr(faiss, 'StandardGpuResources') else 'nem'}")
        if not hasattr(faiss, "StandardGpuResources"):
            print("Megjegyzés              : a faiss-cpu backend működik; GPU FAISS külön platformfüggő csomag.")
    except Exception as exc:
        print(f"Import                  : HIBA - {exc}")

    print("\n[NVIDIA CUDA Python]")
    try:
        from cuda.bindings import driver  # noqa: F401

        print("cuda.bindings import    : OK")
    except Exception as exc:
        print(f"cuda.bindings import    : NEM ELÉRHETŐ - {exc}")

    print("\n[PyTorch CUDA]")
    try:
        import torch

        print(f"torch.__version__       : {torch.__version__}")
        print(f"torch CUDA runtime      : {torch.version.cuda or 'CPU build / nincs CUDA runtime'}")
        print(f"torch.cuda.is_available : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU                     : {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"VRAM                    : {props.total_memory / 1024**3:.2f} GiB")
        else:
            print("Megjegyzés              : a projekt CPU módban továbbra is teljesen használható.")
    except Exception as exc:
        print(f"PyTorch ellenőrzés      : HIBA - {exc}")

    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

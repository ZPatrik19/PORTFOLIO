from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
BUILD_VENV = ROOT / ".venv_build"


def venv_python(venv_path: Path = VENV) -> Path:
    if sys.platform.startswith("win"):
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def run(
    command: list[str],
    *,
    timeout: int | None = None,
) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        timeout=timeout,
    )


def remove_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=False)


def pip_is_working(venv_path: Path) -> bool:
    python = venv_python(venv_path)
    if not python.exists():
        return False

    try:
        result = subprocess.run(
            [str(python), "-m", "pip", "--version"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False

    return result.returncode == 0


def create_stdlib_venv(interpreter: str, target: Path, timeout: int) -> bool:
    try:
        run(
            [interpreter, "-m", "venv", str(target)],
            timeout=timeout,
        )
        return pip_is_working(target)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def ensure_virtualenv(interpreter: str) -> None:
    probe = subprocess.run(
        [interpreter, "-m", "virtualenv", "--version"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    if probe.returncode == 0:
        return

    pip_probe = subprocess.run(
        [interpreter, "-m", "pip", "--version"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    if pip_probe.returncode != 0:
        raise RuntimeError(
            "The launcher Python does not have a working pip installation.\n"
            "Install/reinstall Python with pip enabled, then run setup again."
        )

    run(
        [
            interpreter,
            "-m",
            "pip",
            "install",
            "--user",
            "--upgrade",
            "virtualenv",
        ],
        timeout=300,
    )


def create_virtualenv_fallback(interpreter: str, target: Path, timeout: int) -> bool:
    ensure_virtualenv(interpreter)

    try:
        run(
            [
                interpreter,
                "-m",
                "virtualenv",
                "--python",
                interpreter,
                str(target),
            ],
            timeout=timeout,
        )
        return pip_is_working(target)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def create_environment(
    *,
    recreate: bool,
    interpreter: str,
    timeout: int,
) -> None:
    if recreate:
        remove_dir(VENV)

    if VENV.exists():
        if pip_is_working(VENV):
            print(f"Healthy environment already exists: {VENV}")
            return

        print("Existing .venv is incomplete. Recreating it.")
        remove_dir(VENV)

    remove_dir(BUILD_VENV)

    print("\n[1/7] Creating project virtual environment...")

    success = create_stdlib_venv(
        interpreter,
        BUILD_VENV,
        timeout,
    )

    if not success:
        remove_dir(BUILD_VENV)
        print("stdlib venv failed/stalled; using virtualenv fallback.")

        success = create_virtualenv_fallback(
            interpreter,
            BUILD_VENV,
            timeout,
        )

    if not success:
        remove_dir(BUILD_VENV)
        raise RuntimeError(
            "Could not create a healthy virtual environment."
        )

    BUILD_VENV.rename(VENV)

    if not pip_is_working(VENV):
        raise RuntimeError(
            "Environment was created, but pip verification failed."
        )

    print(f"Environment ready: {VENV}")


def install_dependencies() -> None:
    python = str(venv_python())

    print("\n[2/7] Upgrading packaging tools...")
    run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
        ],
        timeout=600,
    )

    print("\n[3/7] Installing project dependencies...")
    run(
        [
            python,
            "-m",
            "pip",
            "install",
            "-r",
            "requirements.txt",
        ],
        timeout=2400,
    )

    print("\n[4/7] Installing project package in editable mode...")
    run(
        [
            python,
            "-m",
            "pip",
            "install",
            "-e",
            ".",
        ],
        timeout=600,
    )


def create_runtime_directories() -> None:
    print("\n[5/7] Creating runtime directories...")

    directories = [
        ROOT / "01_data" / "raw",
        ROOT / "01_data" / "processed",
        ROOT / "04_results" / "figures",
        ROOT / "04_results" / "metrics",
        ROOT / "04_results" / "models",
        ROOT / "04_results" / "predictions",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def register_jupyter_kernel() -> None:
    python = str(venv_python())

    run(
        [
            python,
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            "dry-bean-classification",
            "--display-name",
            "Python (Dry Bean Classification)",
        ],
        timeout=180,
    )


def validate_environment() -> None:
    python = str(venv_python())

    code = (
        "import numpy, pandas, sklearn, imblearn, torch; "
        "import xgboost, lightgbm, catboost; "
        "from dry_bean.models import BeanMLP; "
        "from dry_bean.constants import FEATURE_COLUMNS; "
        "m = BeanMLP(); "
        "y = m(torch.zeros((2, len(FEATURE_COLUMNS)))); "
        "assert tuple(y.shape) == (2, 7); "
        "print('Environment smoke test: OK')"
    )

    run(
        [python, "-c", code],
        timeout=180,
    )


def ensure_dataset(force: bool) -> None:
    python = str(venv_python())

    print("\n[7/7] Checking/downloading dataset...")

    code = (
        "from dry_bean.data import ensure_data; "
        f"df = ensure_data(force={force!r}); "
        "print(f'Dataset ready: {df.shape[0]} rows x {df.shape[1]} columns')"
    )

    run(
        [python, "-c", code],
        timeout=600,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap the entire Dry Bean project: create .venv, install "
            "dependencies, register Jupyter kernel, validate imports, and "
            "download the dataset if it is missing."
        )
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate .venv.",
    )
    parser.add_argument(
        "--force-data",
        action="store_true",
        help="Force the UCI dataset to be downloaded again.",
    )
    parser.add_argument(
        "--venv-python",
        default=sys.executable,
        help="Python executable used to create .venv.",
    )
    parser.add_argument(
        "--venv-timeout",
        type=int,
        default=300,
        help="Timeout for each environment creation attempt.",
    )

    args = parser.parse_args()

    interpreter = str(
        Path(args.venv_python).expanduser().resolve()
    )

    print("=" * 72)
    print("DRY BEAN CLASSIFICATION — PROJECT SETUP")
    print("=" * 72)
    print(f"Bootstrap interpreter: {interpreter}")

    try:
        create_environment(
            recreate=args.recreate,
            interpreter=interpreter,
            timeout=args.venv_timeout,
        )

        install_dependencies()
        create_runtime_directories()

        print("\n[6/7] Registering Jupyter kernel and validating environment...")
        register_jupyter_kernel()
        validate_environment()

        ensure_dataset(args.force_data)

    except KeyboardInterrupt:
        print(
            "\nSetup interrupted. Rerun the same command; "
            "partial temporary environments will be rebuilt safely."
        )
        raise SystemExit(130)

    print("\n" + "=" * 72)
    print("SETUP COMPLETED SUCCESSFULLY")
    print("=" * 72)
    print(f"Project Python: {venv_python()}")
    print("\nNext command:")
    print("    python run_project.py")
    print("\nOptional quick smoke run:")
    print("    python run_project.py --quick")
    print("\nManual activation is NOT required for run_project.py.")


if __name__ == "__main__":
    main()

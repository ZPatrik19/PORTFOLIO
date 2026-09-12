from __future__ import annotations

"""One-command bootstrap for the Smart Battery Energy Management RL project.

What this script does:
1. validates the host Python version;
2. creates a local .venv if needed;
3. upgrades pip/setuptools/wheel inside that virtual environment;
4. installs the project and development dependencies;
5. optionally installs Stable-Baselines3 and/or CityLearn extras;
6. prepares or downloads the configured dataset when files are missing;
7. registers a dedicated Jupyter kernel;
8. runs a lightweight import + environment smoke test.

The script deliberately uses the virtual environment's Python executable for all
subsequent commands. A child process cannot permanently activate a virtual
environment in the parent shell, so the final activation command is printed for
the user instead.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
MIN_PYTHON = (3, 10)
KERNEL_NAME = "battery-energy-rl"
KERNEL_DISPLAY_NAME = "Python (battery-energy-rl)"


def heading(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def run(cmd: list[str], *, cwd: Path = ROOT, check: bool = True) -> subprocess.CompletedProcess:
    printable = " ".join(f'"{part}"' if " " in part else part for part in cmd)
    print(f"> {printable}")
    return subprocess.run(cmd, cwd=cwd, check=check)


def venv_python(venv_dir: Path = VENV_DIR) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def activation_command(venv_dir: Path = VENV_DIR) -> str:
    if os.name == "nt":
        return rf"{venv_dir}\Scripts\Activate.ps1"
    return f"source {venv_dir}/bin/activate"


def validate_host_python() -> None:
    current = sys.version_info[:2]
    if current < MIN_PYTHON:
        raise SystemExit(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required; "
            f"found {sys.version.split()[0]}."
        )
    print(f"Host Python: {sys.version.split()[0]} ({sys.executable})")


def create_virtual_environment(recreate: bool) -> Path:
    if recreate and VENV_DIR.exists():
        heading("Removing existing virtual environment")
        shutil.rmtree(VENV_DIR)

    if not VENV_DIR.exists():
        heading("Creating .venv")
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    else:
        heading("Using existing .venv")
        print(VENV_DIR)

    py = venv_python()
    if not py.exists():
        raise SystemExit(f"Virtual environment Python was not found: {py}")
    return py


def install_dependencies(py: Path, *, with_sb3: bool, with_citylearn: bool) -> None:
    heading("Installing dependencies")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    extras = ["dev"]
    if with_sb3:
        extras.append("sb3")
    if with_citylearn:
        extras.append("citylearn")

    editable_target = f".[{','.join(extras)}]"
    run([str(py), "-m", "pip", "install", "-e", editable_target])


def prepare_data(py: Path, force_data: bool) -> None:
    heading("Preparing project data")
    code = (
        "from pathlib import Path; "
        "from workflow.shared.utils import ensure_data; "
        f"ensure_data(force={str(force_data)}); "
        "print('Data ready in 01_data/raw and 01_data/processed')"
    )
    run([str(py), "-c", code])


def register_kernel(py: Path) -> None:
    heading("Registering Jupyter kernel")
    run(
        [
            str(py),
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            KERNEL_NAME,
            "--display-name",
            KERNEL_DISPLAY_NAME,
        ]
    )


def smoke_test(py: Path) -> None:
    heading("Running smoke test")
    code = r'''
from workflow.shared.utils import ensure_data, load_config, build_env
import numpy as np
from battery_rl.agents import TabularQLearningAgent

cfg = load_config()
full, train, val, test = ensure_data()
env = build_env(train, cfg, random_start=False)
obs, info = env.reset(seed=42)
next_obs, reward, terminated, truncated, info = env.step(1)

assert len(full) > 0 and len(train) > 0 and len(val) > 0 and len(test) > 0
assert obs.shape == next_obs.shape == (len(env.obs_columns),)
assert np.isfinite(obs).all() and np.isfinite(next_obs).all()
assert isinstance(float(reward), float)
print('Smoke test PASSED')
print('Observation shape:', obs.shape)
print('Action space:', env.action_space)
print('First reward:', float(reward))
'''
    run([str(py), "-c", code])


def print_next_steps(py: Path) -> None:
    heading("Setup complete")
    print("Activate the environment:")
    print(f"  {activation_command()}")
    print("\nThen choose the registered Jupyter kernel:")
    print(f"  {KERNEL_DISPLAY_NAME}")
    print("\nUseful commands:")
    print("  python run_project.py --quick")
    print("  pytest")
    print("  jupyter notebook")
    print("  python tools/set_notebook_language.py --language hu")
    print("  python tools/set_notebook_language.py --language en")
    print(f"\nVirtual-environment Python: {py}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap the Smart Battery Energy Management RL project.")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate .venv before installation.")
    parser.add_argument("--with-sb3", action="store_true", help="Also install Stable-Baselines3 optional dependencies.")
    parser.add_argument("--with-citylearn", action="store_true", help="Also install CityLearn optional dependencies.")
    parser.add_argument("--force-data", action="store_true", help="Regenerate/redownload configured data even if files exist.")
    parser.add_argument("--skip-kernel", action="store_true", help="Do not register a Jupyter kernel.")
    parser.add_argument("--skip-smoke-test", action="store_true", help="Skip the final import/environment smoke test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heading("Smart Battery Energy Management RL — STEP 00 SETUP")
    validate_host_python()
    py = create_virtual_environment(args.recreate)
    install_dependencies(py, with_sb3=args.with_sb3, with_citylearn=args.with_citylearn)
    prepare_data(py, force_data=args.force_data)
    if not args.skip_kernel:
        register_kernel(py)
    if not args.skip_smoke_test:
        smoke_test(py)
    print_next_steps(py)


if __name__ == "__main__":
    main()

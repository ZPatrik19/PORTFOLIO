from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"

STEPS = {
    1: "step01_data_acquisition_validation.py",
    2: "step02_eda.py",
    3: "step03_preprocessing.py",
    4: "step04_baselines.py",
    5: "step05_pytorch_training.py",
    6: "step06_evaluation_error_analysis.py",
    7: "step07_inference.py",
    8: "step08_monitoring_drift.py",
    9: "step09_generate_report.py",
}


def project_python() -> Path:
    if sys.platform.startswith("win"):
        python = VENV / "Scripts" / "python.exe"
    else:
        python = VENV / "bin" / "python"

    if not python.exists():
        raise RuntimeError(
            "Project environment does not exist.\n\n"
            "Run setup first:\n"
            "    python 00_setup_project.py\n"
        )

    return python


def verify_project_ready(python: Path) -> None:
    check_code = (
        "import imblearn, sklearn, torch, pandas, numpy; "
        "from dry_bean.data import ensure_data; "
        "df = ensure_data(force=False); "
        "assert len(df) > 0; "
        "print('Project readiness check: OK')"
    )

    try:
        subprocess.run(
            [str(python), "-c", check_code],
            cwd=ROOT,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "The project environment/data is incomplete.\n\n"
            "Repair it with:\n"
            "    python 00_setup_project.py\n\n"
            "or rebuild completely:\n"
            "    python 00_setup_project.py --recreate"
        ) from exc


def parse_steps(specification: str | None) -> list[int]:
    if not specification:
        return list(STEPS)

    selected: set[int] = set()

    for token in specification.split(","):
        token = token.strip()
        if not token:
            continue

        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))

    invalid = selected - set(STEPS)
    if invalid:
        raise ValueError(
            f"Unknown steps: {sorted(invalid)}. Valid range: 1-9."
        )

    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete Dry Bean ML workflow. "
            "By default all workflow steps are executed in order."
        )
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run lightweight smoke-test versions where supported.",
    )
    parser.add_argument(
        "--steps",
        help="Run only selected steps, e.g. 1-4 or 3,5,7.",
    )

    args = parser.parse_args()

    python = project_python()
    verify_project_ready(python)

    selected_steps = parse_steps(args.steps)

    print("=" * 72)
    print("DRY BEAN CLASSIFICATION — PIPELINE")
    print("=" * 72)
    print(f"Interpreter: {python}")
    print(
        "Execution:",
        " -> ".join(f"STEP {step:02d}" for step in selected_steps),
    )

    for step in selected_steps:
        script = ROOT / "workflow" / STEPS[step]

        command = [
            str(python),
            str(script),
        ]

        if args.quick:
            command.append("--quick")

        print(
            f"\n{'=' * 72}\n"
            f"STEP {step:02d}: {script.name}\n"
            f"{'=' * 72}"
        )

        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
        )

    print("\n" + "=" * 72)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 72)
    print("Results are available under:")
    print("    04_results/figures/")
    print("    04_results/metrics/")
    print("    04_results/models/")
    print("    04_results/predictions/")


if __name__ == "__main__":
    main()

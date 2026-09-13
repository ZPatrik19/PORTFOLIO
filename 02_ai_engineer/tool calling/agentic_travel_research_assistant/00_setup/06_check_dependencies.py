"""Fast dependency check used by SETUP_AND_START_UI.bat.

Exit code 0 means all required distributions already satisfy the project's version
constraints. Exit code 1 means installation is required. Nothing is installed or
updated by this script.
"""
from __future__ import annotations

import re
import sys
from importlib import metadata

# distribution -> (minimum inclusive, maximum exclusive)
REQUIRED = {
    "openai": ("1.100.0", None),
    "pydantic": ("2.7.0", None),
    "python-dotenv": ("1.0.1", None),
    "matplotlib": ("3.8.0", None),
    "pytest": ("8.0.0", None),
    "pytest-cov": ("5.0.0", None),
    "jupyterlab": ("4.0.0", None),
    "nbformat": ("5.10.0", None),
    "pandas": ("2.2.0", None),
    "scikit-learn": ("1.4.0", None),
    "joblib": ("1.3.0", None),
    "streamlit": ("1.63.0", "2.0.0"),
    "plotly": ("6.0.0", "7.0.0"),
}


def _version_tuple(value: str) -> tuple[int, ...]:
    # Good enough for the stable numeric versions required by this project.
    nums = re.findall(r"\d+", value)
    return tuple(int(x) for x in nums[:4])


def _pad(v: tuple[int, ...], n: int = 4) -> tuple[int, ...]:
    return v + (0,) * (n - len(v))


def _compatible(installed: str, minimum: str, maximum: str | None) -> bool:
    current = _pad(_version_tuple(installed))
    if current < _pad(_version_tuple(minimum)):
        return False
    if maximum is not None and current >= _pad(_version_tuple(maximum)):
        return False
    return True


def main() -> int:
    problems: list[str] = []
    print("Dependency check (no installation, no updates)\n")
    for dist, (minimum, maximum) in REQUIRED.items():
        try:
            installed = metadata.version(dist)
        except metadata.PackageNotFoundError:
            problems.append(f"MISSING: {dist} >= {minimum}")
            print(f"[MISSING] {dist}")
            continue
        ok = _compatible(installed, minimum, maximum)
        constraint = f">={minimum}" + (f", <{maximum}" if maximum else "")
        print(f"[{'OK' if ok else 'INCOMPATIBLE'}] {dist} {installed} ({constraint})")
        if not ok:
            problems.append(f"INCOMPATIBLE: {dist} {installed} ({constraint})")

    if problems:
        print("\nInstallation is required because at least one dependency is missing or incompatible.")
        for problem in problems:
            print(" -", problem)
        return 1

    print("\nAll requirements are already satisfied. Package upgrades can be skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

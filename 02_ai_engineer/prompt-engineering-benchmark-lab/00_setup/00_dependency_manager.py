"""Idempotent dependency checker used by bootstrap scripts.

The checker resolves simple ``-r`` includes, compares installed distributions
with requested constraints, and installs only missing/outdated packages.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import subprocess
import sys
from pathlib import Path

try:
    from packaging.requirements import Requirement
except ImportError as exc:  # pragma: no cover - bootstrap handles this case
    raise SystemExit("Install bootstrap helper: python -m pip install 'packaging>=24.0'") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_requirements(path: Path, seen: set[Path] | None = None) -> list[Requirement]:
    """Load requirements recursively, supporting local ``-r file`` includes."""
    path = path.resolve()
    seen = seen or set()
    if path in seen:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Requirements file does not exist: {path}")
    seen.add(path)
    requirements: list[Requirement] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r ") or line.startswith("--requirement "):
            included = line.split(maxsplit=1)[1]
            requirements.extend(load_requirements(path.parent / included, seen))
            continue
        requirement = Requirement(line)
        if requirement.marker is None or requirement.marker.evaluate():
            requirements.append(requirement)
    # Preserve first occurrence while avoiding duplicate package work.
    unique: dict[str, Requirement] = {}
    for requirement in requirements:
        unique[requirement.name.lower()] = requirement
    return list(unique.values())


def installed_version(distribution_name: str) -> str | None:
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return None


def requirements_needing_install(requirements: list[Requirement]) -> list[Requirement]:
    pending: list[Requirement] = []
    for requirement in requirements:
        version = installed_version(requirement.name)
        if version is None or (requirement.specifier and version not in requirement.specifier):
            pending.append(requirement)
    return pending


def print_status(requirements: list[Requirement]) -> None:
    print("\nDependency status")
    print("-" * 78)
    for requirement in requirements:
        version = installed_version(requirement.name)
        if version is None:
            status = "MISSING"
        elif requirement.specifier and version not in requirement.specifier:
            status = "UPDATE NEEDED"
        else:
            status = "OK"
        requested = str(requirement.specifier) or "any"
        print(f"{requirement.name:<25} installed={version or '-':<15} required={requested:<16} {status}")


def install_requirements(requirements: list[Requirement]) -> None:
    if not requirements:
        print("\nDependencies already satisfy constraints. Nothing to install.")
        return
    print("\nInstalling/updating only packages that need attention:")
    for requirement in requirements:
        print(f"  - {requirement}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--upgrade-strategy",
            "only-if-needed",
            *[str(requirement) for requirement in requirements],
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def run_pip_check() -> None:
    subprocess.run([sys.executable, "-m", "pip", "check"], cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check and repair project dependencies.")
    parser.add_argument("--requirements", default="requirements-dev.txt", help="Requirements file relative to repo root.")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if not ((3, 10) <= sys.version_info < (3, 15)):
        raise SystemExit(
            f"Python {sys.version_info.major}.{sys.version_info.minor} is unsupported; "
            "use Python 3.10 through 3.14."
        )

    requirements = load_requirements(PROJECT_ROOT / args.requirements)
    print_status(requirements)
    pending = requirements_needing_install(requirements)
    if args.check_only:
        if pending:
            raise SystemExit(2)
        run_pip_check()
        return
    install_requirements(pending)
    run_pip_check()
    print("\nDependency check complete.")


if __name__ == "__main__":
    main()

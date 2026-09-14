"""Portable project path discovery and repository-relative paths.

The application is repository-backed: datasets, configs, notebooks and output
folders intentionally live outside the installed Python package.  Therefore the
project root must be discovered explicitly rather than inferred from the
``site-packages`` location of an installed wheel.

Discovery order:
1. ``PROMPT_BENCHMARK_ROOT`` environment variable.
2. Current working directory and its parents.
3. Source-file location and its parents (covers editable installs).

A clear error is raised if no repository root can be found. Docker sets
``PROMPT_BENCHMARK_ROOT=/app`` so non-editable package installs remain correct.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT_ENV_VAR = "PROMPT_BENCHMARK_ROOT"
_REQUIRED_ROOT_MARKERS = ("pyproject.toml", "configs", "01_data", "05_scripts")


class ProjectRootError(RuntimeError):
    """Raised when the repository root cannot be resolved safely."""


def _looks_like_project_root(path: Path) -> bool:
    return all((path / marker).exists() for marker in _REQUIRED_ROOT_MARKERS)


def _candidate_roots(start: Path) -> list[Path]:
    resolved = start.resolve()
    if resolved.is_file():
        resolved = resolved.parent
    return [resolved, *resolved.parents]


def discover_project_root() -> Path:
    """Resolve the repository root without assuming editable installation.

    ``PROMPT_BENCHMARK_ROOT`` is authoritative when present and must point to a
    valid repository. Otherwise the current working directory is searched first
    because packaged/containerized execution keeps the repo as runtime data.
    Finally the source location is searched, which handles normal editable
    development installs.
    """
    configured = os.getenv(ROOT_ENV_VAR, "").strip()
    if configured:
        configured_path = Path(configured).expanduser().resolve()
        if not _looks_like_project_root(configured_path):
            raise ProjectRootError(
                f"{ROOT_ENV_VAR}={configured_path} is not a valid project root; "
                f"expected markers: {', '.join(_REQUIRED_ROOT_MARKERS)}"
            )
        return configured_path

    checked: list[Path] = []
    for start in (Path.cwd(), Path(__file__)):
        for candidate in _candidate_roots(start):
            if candidate in checked:
                continue
            checked.append(candidate)
            if _looks_like_project_root(candidate):
                return candidate

    checked_text = ", ".join(str(path) for path in checked[:8])
    raise ProjectRootError(
        "Could not locate the Prompt Engineering Benchmark repository root. "
        f"Run the command from the repository or set {ROOT_ENV_VAR}. "
        f"Checked: {checked_text}"
    )


PROJECT_ROOT = discover_project_root()


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical repository paths used by scripts, UI and package modules."""

    root: Path = field(default_factory=lambda: PROJECT_ROOT)

    @property
    def setup(self) -> Path:
        return self.root / "00_setup"

    @property
    def data(self) -> Path:
        return self.root / "01_data"

    @property
    def processed_data(self) -> Path:
        return self.data / "processed"

    @property
    def mock_data(self) -> Path:
        return self.data / "mock"

    @property
    def benchmark_suites(self) -> Path:
        return self.data / "benchmark_suites"

    @property
    def notebooks(self) -> Path:
        return self.root / "02_notebooks"

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def prompts(self) -> Path:
        return self.configs / "prompts"

    @property
    def prompt_examples(self) -> Path:
        return self.prompts / "examples"

    @property
    def custom_prompts(self) -> Path:
        return self.prompts / "custom"

    @property
    def playground_prompts(self) -> Path:
        return self.prompts / "playground"

    @property
    def outputs(self) -> Path:
        return self.root / "07_outputs"

    @property
    def results(self) -> Path:
        return self.outputs / "results"

    @property
    def reports(self) -> Path:
        return self.outputs / "reports"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    @property
    def logs(self) -> Path:
        return self.outputs / "logs"

    @property
    def docs(self) -> Path:
        return self.root / "docs"

    @property
    def deployment(self) -> Path:
        return self.root / "06_deployment"


PATHS = ProjectPaths()

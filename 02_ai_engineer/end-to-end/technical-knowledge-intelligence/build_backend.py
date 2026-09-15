"""Small self-contained PEP 517/660 backend for deterministic local/CI installs.

The project intentionally keeps the backend dependency-free. This makes
``pip install -e . --no-build-isolation`` work in clean Python environments
where setuptools is not preinstalled (notably Python 3.12+ CI runners).
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    tomllib = None

ROOT = Path(__file__).resolve().parent
PYPROJECT = ROOT / "pyproject.toml"
PACKAGE_ROOT = ROOT / "03_pipeline"
PACKAGE_DIR = PACKAGE_ROOT / "tkip"


def _project() -> dict:
    if tomllib is not None:
        return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    # Python 3.10 fallback. Keep only the metadata needed to build/install the
    # package; supported launchers normally select Python 3.11+ first.
    return {
        "name": "technical-knowledge-intelligence",
        "version": "1.1.1",
        "description": "Production-oriented technical knowledge intelligence platform",
        "requires-python": ">=3.10",
        "dependencies": [],
        "optional-dependencies": {},
    }


def _dist_name() -> str:
    return str(_project()["name"]).replace("-", "_")


def _version() -> str:
    return str(_project()["version"])


def _dist_info() -> str:
    return f"{_dist_name()}-{_version()}.dist-info"


def _wheel_name() -> str:
    return f"{_dist_name()}-{_version()}-py3-none-any.whl"


def _metadata_text() -> str:
    project = _project()
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {project['name']}",
        f"Version: {project['version']}",
        f"Summary: {project.get('description', '')}",
        f"Requires-Python: {project.get('requires-python', '>=3.10')}",
    ]
    license_value = project.get("license")
    if isinstance(license_value, dict) and license_value.get("text"):
        lines.append(f"License: {license_value['text']}")
    for requirement in project.get("dependencies", []):
        lines.append(f"Requires-Dist: {requirement}")
    for extra, requirements in project.get("optional-dependencies", {}).items():
        lines.append(f"Provides-Extra: {extra}")
        for requirement in requirements:
            lines.append(f'Requires-Dist: {requirement}; extra == "{extra}"')
    lines.append("")
    return "\n".join(lines)


def _wheel_text() -> str:
    return "\n".join(
        [
            "Wheel-Version: 1.0",
            "Generator: tki-build-backend 1.0",
            "Root-Is-Purelib: true",
            "Tag: py3-none-any",
            "",
        ]
    )


def _entry_points_text() -> str:
    return "[console_scripts]\ntki = tkip.cli:main\n"


def _record_row(path: str, data: bytes) -> tuple[str, str, str]:
    digest = hashlib.sha256(data).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return path, f"sha256={encoded}", str(len(data))


def _write_wheel(wheel_path: Path, *, editable: bool) -> None:
    entries: list[tuple[str, bytes]] = []
    dist_info = _dist_info()

    if editable:
        pth_name = f"{_dist_name()}-editable.pth"
        pth_content = (str(PACKAGE_ROOT.resolve()) + os.linesep).encode("utf-8")
        entries.append((pth_name, pth_content))
    else:
        for source in sorted(PACKAGE_DIR.rglob("*")):
            if source.is_file() and "__pycache__" not in source.parts:
                relative = source.relative_to(PACKAGE_ROOT).as_posix()
                entries.append((relative, source.read_bytes()))

    metadata = _metadata_text().encode("utf-8")
    wheel = _wheel_text().encode("utf-8")
    entry_points = _entry_points_text().encode("utf-8")
    entries.extend(
        [
            (f"{dist_info}/METADATA", metadata),
            (f"{dist_info}/WHEEL", wheel),
            (f"{dist_info}/entry_points.txt", entry_points),
        ]
    )

    record_buffer = io.StringIO(newline="")
    writer = csv.writer(record_buffer, lineterminator="\n")
    for path, data in entries:
        writer.writerow(_record_row(path, data))
    writer.writerow((f"{dist_info}/RECORD", "", ""))
    entries.append((f"{dist_info}/RECORD", record_buffer.getvalue().encode("utf-8")))

    wheel_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, data in entries:
            archive.writestr(path, data)


def get_requires_for_build_wheel(config_settings=None) -> list[str]:
    return []


def get_requires_for_build_editable(config_settings=None) -> list[str]:
    return []


def get_requires_for_build_sdist(config_settings=None) -> list[str]:
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None) -> str:
    return _prepare_metadata(metadata_directory)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None) -> str:
    return _prepare_metadata(metadata_directory)


def _prepare_metadata(metadata_directory: str) -> str:
    dist_info = _dist_info()
    target = Path(metadata_directory) / dist_info
    target.mkdir(parents=True, exist_ok=True)
    (target / "METADATA").write_text(_metadata_text(), encoding="utf-8")
    (target / "WHEEL").write_text(_wheel_text(), encoding="utf-8")
    (target / "entry_points.txt").write_text(_entry_points_text(), encoding="utf-8")
    return dist_info


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None) -> str:
    name = _wheel_name()
    _write_wheel(Path(wheel_directory) / name, editable=False)
    return name


def build_editable(wheel_directory, config_settings=None, metadata_directory=None) -> str:
    name = _wheel_name()
    _write_wheel(Path(wheel_directory) / name, editable=True)
    return name


def build_sdist(sdist_directory, config_settings=None) -> str:
    project_name = str(_project()["name"])
    base_name = f"{project_name}-{_version()}"
    output_name = f"{base_name}.tar.gz"
    output_path = Path(sdist_directory) / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / base_name
        shutil.copytree(
            ROOT,
            stage,
            ignore=shutil.ignore_patterns(
                ".git",
                ".venv",
                "__pycache__",
                ".pytest_cache",
                ".mypy_cache",
                ".ruff_cache",
            ),
        )
        with tarfile.open(output_path, "w:gz") as archive:
            archive.add(stage, arcname=base_name)
    return output_name

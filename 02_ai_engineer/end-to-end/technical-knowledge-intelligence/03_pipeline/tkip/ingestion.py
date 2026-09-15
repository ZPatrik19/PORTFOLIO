"""Corpus discovery and stable document manifest/version management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import load_config, resolve_path
from .logging_config import get_logger
from .models import DocumentRecord
from .utils import sha256_file, stable_id

LOGGER = get_logger(__name__)
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown", ".html", ".htm", ".epub"}
IGNORED_FILENAMES = {"readme.md", "readme.txt"}


def discover_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for base_path in paths:
        if not base_path.exists():
            continue
        for candidate in base_path.rglob("*"):
            if (
                candidate.is_file()
                and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
                and candidate.name.lower() not in IGNORED_FILENAMES
            ):
                files.append(candidate)
    return sorted(files)


def build_manifest(config: dict[str, Any] | None = None) -> list[DocumentRecord]:
    cfg = config or load_config()
    user_root = resolve_path(cfg["paths"]["user_library"])
    reference_root = resolve_path(cfg["paths"]["reference_docs"])
    project_root = Path(cfg["_project_root"])
    manifest_path = resolve_path(cfg["paths"]["processed"]) / "document_manifest.json"
    previous = _load_previous_manifest(manifest_path)

    records: list[DocumentRecord] = []
    for path in discover_files([user_root, reference_root]):
        relative_path = path.relative_to(project_root)
        source_type = "private" if user_root in path.parents else "public"
        record = DocumentRecord(
            document_id=stable_id(str(relative_path).lower(), prefix="doc"),
            filename=path.name,
            title=path.stem.replace("_", " ").replace("-", " ").title(),
            document_type=path.suffix.lower().lstrip("."),
            source=str(relative_path),
            source_type=source_type,
            checksum=sha256_file(path),
            path=str(path),
        )
        old = previous.get(record.document_id)
        if old:
            old_version = int(old.get("document_version", "1"))
            record.document_version = str(
                old_version if old.get("checksum") == record.checksum else old_version + 1
            )
        records.append(record)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps([record.model_dump() for record in records], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return records


def changed_documents(records: list[DocumentRecord], indexed_state_path: Path) -> list[DocumentRecord]:
    indexed_state = _read_json_dict(indexed_state_path)
    return [record for record in records if indexed_state.get(record.document_id) != record.checksum]


def _load_previous_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        return {row["document_id"]: row for row in rows}
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        LOGGER.warning("Ignoring invalid previous manifest %s: %s", path, exc)
        return {}


def _read_json_dict(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Ignoring invalid indexed state %s: %s", path, exc)
        return {}

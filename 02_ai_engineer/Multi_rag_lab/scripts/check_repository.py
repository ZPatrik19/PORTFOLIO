from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DIRS = {
    "rag_engine",
    "ui",
    "config",
    "data",
    "artifacts",
    "scripts",
    "tests",
    "docs",
}
FORBIDDEN_DIR_NAMES = {"__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "src"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo"}
GENERATED_PREFIXES = (
    "artifacts/indexes/",
    "artifacts/evaluations/",
    "artifacts/experiments/",
    "artifacts/models/",
    "data/raw/",
    "data/processed/",
)
ALLOWED_GENERATED_NAMES = {".gitkeep", "README.md"}
WINDOWS_ABSOLUTE_PATH = re.compile(r"(?i)(?:[A-Z]:\\(?:Users|Projects|Program Files|Windows)\\)")


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _tracked_files() -> list[Path] | None:
    if not (ROOT / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return [ROOT / raw.decode("utf-8") for raw in result.stdout.split(b"\0") if raw]


def main() -> int:
    errors: list[str] = []

    for name in sorted(EXPECTED_DIRS):
        if not (ROOT / name).is_dir():
            errors.append(f"missing required directory: {name}")

    if (ROOT / "rag_lab").exists():
        errors.append("legacy rag_lab/ package must not return")

    tracked = _tracked_files()
    if tracked is not None:
        for path in tracked:
            rel = path.relative_to(ROOT)
            parts = set(rel.parts)
            if parts & FORBIDDEN_DIR_NAMES:
                errors.append(f"generated/legacy path committed: {rel.as_posix()}")
            if any(part.endswith(".egg-info") for part in rel.parts):
                errors.append(f"generated package metadata committed: {rel.as_posix()}")
            if path.suffix in FORBIDDEN_SUFFIXES:
                errors.append(f"compiled Python file committed: {rel.as_posix()}")
            rel_text = rel.as_posix()
            if rel_text.startswith(GENERATED_PREFIXES) and path.name not in ALLOWED_GENERATED_NAMES:
                errors.append(f"generated runtime artifact committed: {rel_text}")

    for path in (ROOT / "rag_engine").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "import streamlit" in text or "from streamlit" in text:
            errors.append(f"core/UI boundary violation (Streamlit import): {_relative(path)}")
        if re.search(r"(^|\n)\s*(?:from\s+ui\b|import\s+ui\b)", text):
            errors.append(f"core/UI boundary violation (ui import): {_relative(path)}")

    text_extensions = {".py", ".md", ".toml", ".yaml", ".yml", ".json", ".bat", ".sh"}
    candidates = tracked if tracked is not None else [p for p in ROOT.rglob("*") if p.is_file()]
    for path in candidates:
        if path.suffix.lower() not in text_extensions:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if WINDOWS_ABSOLUTE_PATH.search(text):
            errors.append(f"machine-specific absolute Windows path: {_relative(path)}")

    if errors:
        print("Repository contract FAILED:")
        for error in sorted(set(errors)):
            print(f" - {error}")
        return 1

    print("Repository contract OK")
    print(" - expected top-level domains present")
    print(" - tracked repository files respect hygiene rules")
    print(" - generated data/artifacts are not part of the release layout")
    print(" - rag_engine has no Streamlit/UI dependency")
    print(" - no machine-specific Windows paths found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

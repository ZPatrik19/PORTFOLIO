"""Validate the local Python/runtime environment without making network calls."""

from __future__ import annotations

import importlib.util
import platform
import shutil
import sys

REQUIRED_MODULES = (
    "tkip",
    "pydantic",
    "yaml",
    "fastapi",
    "uvicorn",
    "streamlit",
    "numpy",
    "pandas",
    "pymupdf",
    "bs4",
    "requests",
    "dotenv",
)
OPTIONAL_MODULES = ("graphviz", "google.genai", "qdrant_client", "langchain_core")


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def main() -> int:
    print("Python:", sys.version)
    print("Platform:", platform.platform())
    missing: list[str] = []
    for module in REQUIRED_MODULES:
        available = has_module(module)
        print(f"{module:24} {'OK' if available else 'MISSING'}")
        if not available:
            missing.append(module)

    print("\nOptional capabilities:")
    for module in OPTIONAL_MODULES:
        print(f"{module:24} {'OK' if has_module(module) else 'OPTIONAL/MISSING'}")
    print(f"{'Graphviz dot executable':24} {shutil.which('dot') or 'OPTIONAL/MISSING'}")

    if missing:
        print("\n[ERROR] Missing required modules:", ", ".join(missing))
        return 1
    print("\n[OK] Runtime environment is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

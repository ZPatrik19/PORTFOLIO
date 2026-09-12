from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent
WORKFLOW = ROOT / "workflow"


def status() -> None:
    notebooks = [ROOT / "00_setup_project.ipynb", *sorted(WORKFLOW.glob("*.ipynb"))]
    notebooks = [path for path in notebooks if path.exists()]
    if not notebooks:
        print("No workflow notebooks found.")
        return
    states = {}
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        language = notebook.metadata.get("i18n", {}).get("active_language", "unknown")
        states.setdefault(language, []).append(path.name)
    for language, names in states.items():
        print(f"{language}: {len(names)} notebook(s)")
        for name in names:
            print(f"  - {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Switch or inspect notebook documentation language")
    parser.add_argument("--language", choices=["hu", "en"])
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        status()
        return
    if not args.language:
        parser.error("Use --language hu|en or --status")
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "set_notebook_language.py"), "--language", args.language],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()

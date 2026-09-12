from __future__ import annotations

"""Switch notebook narrative language without duplicating notebook logic.

Each translatable cell stores both variants under:

    cell.metadata.i18n.hu
    cell.metadata.i18n.en

The active ``cell.source`` is replaced in-place. Code, execution counts, outputs,
and embedded figures are preserved. The mechanism is generic and also supports
code cells if language-specific code/comment variants are added later.
"""

import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "workflow"
LANGUAGES = {"hu": "Hungarian", "en": "English"}


def notebook_paths(base: Path = WORKFLOW) -> list[Path]:
    return sorted(base.glob("step*.ipynb"))


def source_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def switch_notebook(path: Path, language: str) -> tuple[int, int]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    translated = 0
    missing = 0

    for cell in nb.get("cells", []):
        variants = cell.get("metadata", {}).get("i18n")
        if not variants:
            continue
        if language not in variants:
            missing += 1
            continue
        cell["source"] = source_lines(str(variants[language]))
        translated += 1

    nb.setdefault("metadata", {})["project_language"] = language
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return translated, missing


def switch_all(language: str, base: Path = WORKFLOW) -> None:
    paths = notebook_paths(base)
    if not paths:
        raise SystemExit(f"No workflow notebooks found under {base}")

    total = 0
    total_missing = 0
    for path in paths:
        changed, missing = switch_notebook(path, language)
        total += changed
        total_missing += missing
        print(f"{path.name}: {changed} translated cells")

    marker = ROOT / ".notebook_language"
    marker.write_text(language + "\n", encoding="utf-8")
    print(f"\nActive notebook language: {LANGUAGES[language]} ({language})")
    print(f"Translated cells: {total}")
    if total_missing:
        print(f"WARNING: {total_missing} cells had i18n metadata but no {language!r} variant.")
    print("Notebook outputs and embedded figures were preserved.")


def status() -> None:
    rows = []
    for path in notebook_paths():
        nb = json.loads(path.read_text(encoding="utf-8"))
        active = nb.get("metadata", {}).get("project_language", "unknown")
        translatable = sum(bool(c.get("metadata", {}).get("i18n")) for c in nb.get("cells", []))
        rows.append((path.name, active, translatable))
    for name, active, count in rows:
        print(f"{name:<48} language={active:<7} translatable_cells={count}")


def export_both(destination: Path) -> None:
    destination = destination.resolve()
    if destination.exists():
        shutil.rmtree(destination)
    for language in LANGUAGES:
        target = destination / language
        target.mkdir(parents=True, exist_ok=True)
        for src in notebook_paths():
            dst = target / src.name
            shutil.copy2(src, dst)
            switch_notebook(dst, language)
        print(f"Exported {LANGUAGES[language]} notebooks to {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Switch HU/EN markdown (and future language-aware code comments) in workflow notebooks."
    )
    parser.add_argument("--language", "-l", choices=sorted(LANGUAGES), help="Activate Hungarian (hu) or English (en).")
    parser.add_argument("--status", action="store_true", help="Show current language metadata for every notebook.")
    parser.add_argument(
        "--export-both",
        metavar="DIR",
        type=Path,
        help="Create generated HU and EN notebook copies under DIR without changing the canonical source-of-truth metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.status:
        status()
        return
    if args.export_both:
        export_both(args.export_both)
        return
    if args.language:
        switch_all(args.language)
        return
    raise SystemExit("Choose --language hu|en, --status, or --export-both DIR.")


if __name__ == "__main__":
    main()

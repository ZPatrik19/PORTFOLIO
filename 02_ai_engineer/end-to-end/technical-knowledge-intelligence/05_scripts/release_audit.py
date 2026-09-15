"""Repository release audit used before publishing a portfolio build."""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = {
    "Gemini/authorization-looking key": re.compile(r"\bAQ\.[A-Za-z0-9_-]{20,}\b"),
    "Google legacy API key": re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "Windows user path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+"),
}
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".bat", ".sh", ".json", ".example"}


@dataclass
class AuditReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)

    def check(self, message: str) -> None:
        self.checks.append(message)


def iter_repo_files() -> Iterable[Path]:
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def audit_python(report: AuditReport) -> None:
    count = 0
    for path in iter_repo_files():
        if path.suffix != ".py":
            continue
        count += 1
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            report.errors.append(f"Python parse failure: {path.relative_to(ROOT)}: {exc}")
    report.check(f"Python syntax parsed: {count} files")


def audit_notebooks(report: AuditReport) -> None:
    notebooks = list((ROOT / "02_notebooks").glob("*.ipynb"))
    missing_ids = 0
    for path in notebooks:
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            report.errors.append(f"Notebook parse failure: {path.name}: {exc}")
            continue
        if not notebook.get("cells"):
            report.errors.append(f"Notebook has no cells: {path.name}")
        missing_ids += sum(1 for cell in notebook.get("cells", []) if not cell.get("id"))
    if missing_ids:
        report.warnings.append(f"Notebook cells without stable id: {missing_ids}")
    report.check(f"Notebook JSON validated: {len(notebooks)} notebooks")


def audit_yaml(report: AuditReport) -> None:
    count = 0
    for path in iter_repo_files():
        if path.suffix not in {".yaml", ".yml"}:
            continue
        count += 1
        try:
            list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
        except (OSError, yaml.YAMLError) as exc:
            report.errors.append(f"YAML parse failure: {path.relative_to(ROOT)}: {exc}")
    report.check(f"YAML parsed: {count} files")


def audit_secrets_and_paths(report: AuditReport) -> None:
    hits: list[str] = []
    for path in iter_repo_files():
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"Dockerfile", ".gitignore"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                # Example placeholders are intentionally permitted.
                if path.name in {"secret.example.yaml", ".env.example"} and "replace_me" in text:
                    continue
                hits.append(f"{name}: {path.relative_to(ROOT)}")
    report.errors.extend(f"Potential secret/path leak: {hit}" for hit in hits)
    report.check(f"Secret/absolute-path scan: {len(hits)} suspicious hit(s)")


def audit_git_hygiene(report: AuditReport) -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    required = [".venv/", ".env", "__pycache__", ".pytest_cache", "01_data/user_library"]
    missing = [entry for entry in required if entry not in gitignore]
    if missing:
        report.errors.append(f".gitignore missing required entries: {missing}")
    report.check("Repository hygiene rules checked")



def audit_architecture_hygiene(report: AuditReport) -> None:
    """Reject runtime import hacks and silent exception swallowing in source code."""

    import_hack_pattern = re.compile(r"sys\.path\.(?:insert|append)|PYTHONPATH=03_pipeline")
    silent_exception_pattern = re.compile(
        r"except(?:\s+Exception)?\s*:\s*(?:#.*\n\s*)?pass\b",
        re.MULTILINE,
    )
    import_hacks: list[str] = []
    silent_exceptions: list[str] = []
    source_roots = ["00_setup", "03_pipeline", "04_api", "05_scripts", "05_ui"]
    for root_name in source_roots:
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if import_hack_pattern.search(text):
                import_hacks.append(str(path.relative_to(ROOT)))
            if silent_exception_pattern.search(text):
                silent_exceptions.append(str(path.relative_to(ROOT)))

    if import_hacks:
        report.errors.append(f"Runtime import hacks remain: {sorted(import_hacks)}")
    if silent_exceptions:
        report.errors.append(f"Silent exception swallowing remains: {sorted(silent_exceptions)}")
    report.check(
        f"Architecture hygiene: {len(import_hacks)} import hack(s), "
        f"{len(silent_exceptions)} silent exception handler(s)"
    )


def audit_required_release_files(report: AuditReport) -> None:
    """Verify that the portfolio/release documentation and runners exist."""

    required = [
        "README.md",
        "09_docs/06_TESTING.md",
        "FINAL_VALIDATION_REPORT.md",
        "pyproject.toml",
        "requirements.txt",
        ".env.example",
        ".gitignore",
        ".gitattributes",
        "Dockerfile",
        "docker-compose.yml",
        "setup.bat",
        "setup.sh",
        "run_project.bat",
        "run_project.sh",
        "09_docs/03_DATA_PIPELINE.md",
        "09_docs/PROJECT_STORY.md",
        "10_deployment/kubernetes/api-deployment.yaml",
        "10_deployment/kubernetes/ui-deployment.yaml",
        ".github/workflows/ci.yml",
    ]
    missing = [item for item in required if not (ROOT / item).exists()]
    if missing:
        report.errors.append(f"Required release files missing: {missing}")
    report.check(f"Required release files checked: {len(required)} entries")


def audit_launcher_integrity(report: AuditReport) -> None:
    """Validate platform line endings and Windows internal batch labels."""

    batch_files = sorted(ROOT.glob("*.bat"))
    shell_files = sorted(ROOT.glob("*.sh"))

    for path in batch_files:
        raw = path.read_bytes()
        if not raw or raw.count(b"\n") != raw.count(b"\r\n"):
            report.errors.append(
                f"Windows batch file must use CRLF only: {path.relative_to(ROOT)}"
            )

    for path in shell_files:
        raw = path.read_bytes()
        if b"\r\n" in raw:
            report.errors.append(
                f"Unix shell file must use LF only: {path.relative_to(ROOT)}"
            )

    windows_runner = ROOT / "run_project.bat"
    if windows_runner.exists():
        script = windows_runner.read_text(encoding="utf-8")
        labels = set(re.findall(r"(?im)^\s*:([A-Za-z0-9_-]+)\s*$", script))
        references = set(re.findall(r"(?i)\bcall\s+:([A-Za-z0-9_-]+)", script))
        references.update(re.findall(r"(?i)\bgoto\s+([A-Za-z0-9_-]+)", script))
        missing = sorted(references - labels)
        if missing:
            report.errors.append(f"Windows runner references missing labels: {missing}")

    attributes = ROOT / ".gitattributes"
    if not attributes.exists():
        report.errors.append("Missing .gitattributes line-ending policy")
    else:
        rules = attributes.read_text(encoding="utf-8")
        if "*.bat text eol=crlf" not in rules or "*.sh  text eol=lf" not in rules:
            report.errors.append(".gitattributes does not enforce launcher line endings")

    report.check(
        f"Launcher integrity checked: {len(batch_files)} batch file(s), "
        f"{len(shell_files)} shell file(s)"
    )

def audit_tools(report: AuditReport) -> None:
    for command in ("docker", "kubectl", "dot"):
        value = shutil.which(command)
        if value:
            report.check(f"External tool available: {command} -> {value}")
        else:
            report.warnings.append(f"External tool not available in this environment: {command}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output")
    args = parser.parse_args()

    report = AuditReport()
    audit_python(report)
    audit_notebooks(report)
    audit_yaml(report)
    audit_secrets_and_paths(report)
    audit_git_hygiene(report)
    audit_architecture_hygiene(report)
    audit_required_release_files(report)
    audit_launcher_integrity(report)
    audit_tools(report)

    payload = {"checks": report.checks, "warnings": report.warnings, "errors": report.errors}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for message in report.checks:
            print(f"[PASS] {message}")
        for message in report.warnings:
            print(f"[WARN] {message}")
        for message in report.errors:
            print(f"[FAIL] {message}")
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())

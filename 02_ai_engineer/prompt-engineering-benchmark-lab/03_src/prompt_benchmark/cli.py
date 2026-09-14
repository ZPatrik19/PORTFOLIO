"""Command-line entry point for common project workflows.

The CLI deliberately forwards provider/workflow-specific flags unchanged to the
existing scripts.  This keeps one source of truth for benchmark arguments while
still providing a convenient installed console command.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from prompt_benchmark.paths import PATHS

SCRIPT_MAP = {
    "prepare-data": "02_prepare_data.py",
    "benchmark": "04_run_benchmark.py",
    "ablation": "05_run_ablation.py",
    "parameter-sweep": "06_run_parameter_sweep.py",
    "report": "07_generate_report.py",
    "smoke": "09_run_smoke_test.py",
}
COMMANDS = (*SCRIPT_MAP.keys(), "ui")


def _run_script(name: str, extra_args: list[str]) -> int:
    script = PATHS.root / "05_scripts" / SCRIPT_MAP[name]
    return subprocess.run(
        [sys.executable, str(script), *extra_args],
        cwd=PATHS.root,
        check=False,
    ).returncode


def _run_ui(extra_args: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="prompt-benchmark ui", description="Start the Streamlit application.")
    parser.add_argument("--port", type=int, default=8501)
    parsed = parser.parse_args(extra_args)
    script = PATHS.root / "05_scripts" / "11_ui_app.py"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(script),
            "--server.address=0.0.0.0",
            f"--server.port={parsed.port}",
        ],
        cwd=PATHS.root,
        check=False,
    ).returncode


def _print_help() -> None:
    parser = argparse.ArgumentParser(prog="prompt-benchmark", description="Prompt Engineering Benchmark Lab CLI")
    parser.add_argument("command", nargs="?", choices=COMMANDS)
    parser.epilog = (
        "Workflow-specific options are forwarded unchanged. Examples:\n"
        "  prompt-benchmark prepare-data --source mock\n"
        "  prompt-benchmark benchmark --provider mock --strategy p0_zero_shot --limit 12\n"
        "  prompt-benchmark report --provider mock\n"
        "  prompt-benchmark ui --port 8501"
    )
    parser.print_help()


def main(argv: list[str] | None = None) -> None:
    """Dispatch one CLI command and preserve the child workflow exit status."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_help()
        raise SystemExit(0)

    command, *extra_args = args
    if command not in COMMANDS:
        print(f"Unknown command: {command}\n", file=sys.stderr)
        _print_help()
        raise SystemExit(2)

    code = _run_ui(extra_args) if command == "ui" else _run_script(command, extra_args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()

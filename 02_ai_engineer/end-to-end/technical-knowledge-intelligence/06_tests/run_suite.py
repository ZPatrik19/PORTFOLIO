"""Cross-platform test-suite launcher.

Examples:
    python 06_tests/run_suite.py quick
    python 06_tests/run_suite.py offline
    python 06_tests/run_suite.py evaluation
    python 06_tests/run_suite.py live
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class TestProfile:
    description: str
    pytest_args: tuple[str, ...]


PROFILES: dict[str, TestProfile] = {
    "quick": TestProfile(
        "Fast developer check: unit + smoke tests.",
        ("06_tests/unit", "06_tests/smoke"),
    ),
    "offline": TestProfile(
        "Recommended default: all deterministic suites except performance and live Gemini.",
        (
            "06_tests/unit",
            "06_tests/integration",
            "06_tests/evaluation",
            "06_tests/regression",
            "06_tests/robustness",
            "06_tests/smoke",
        ),
    ),
    "unit": TestProfile("Unit tests only.", ("06_tests/unit",)),
    "integration": TestProfile("Integration tests only.", ("06_tests/integration",)),
    "evaluation": TestProfile("AI/retrieval metric validation only.", ("06_tests/evaluation",)),
    "regression": TestProfile("Golden regression tests only.", ("06_tests/regression",)),
    "robustness": TestProfile(
        "Prompt-injection and unusual-input tests.", ("06_tests/robustness",)
    ),
    "performance": TestProfile("Local performance guardrails.", ("06_tests/performance",)),
    "smoke": TestProfile("API/application smoke tests.", ("06_tests/smoke",)),
    "all-offline": TestProfile(
        "Every local/offline suite, including performance tests.",
        ("06_tests", "-m", "not live_gemini"),
    ),
    "live": TestProfile(
        "Explicit live Gemini smoke test. Consumes API quota.",
        ("06_tests/live_gemini",),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a named Technical Knowledge Intelligence test profile."
    )
    parser.add_argument("profile", choices=sorted(PROFILES), nargs="?", default="offline")
    parser.add_argument("--verbose", action="store_true", help="Use verbose pytest output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = PROFILES[args.profile]

    print(f"\nTKI test profile: {args.profile}")
    print(f"Purpose: {profile.description}\n")

    if args.profile == "live":
        os.environ["TKI_RUN_LIVE_GEMINI"] = "1"
        print("WARNING: live profile makes real Gemini API requests and consumes quota.\n")

    pytest_args = [*profile.pytest_args, "--strict-markers", "-ra"]
    pytest_args.append("-vv" if args.verbose else "-q")
    return int(pytest.main(pytest_args))


if __name__ == "__main__":
    sys.exit(main())

"""Input/output guardrails for document exfiltration and quote limits."""

from __future__ import annotations

import re
from typing import Any

EXFILTRATION_PATTERNS = (
    r"reveal the entire book",
    r"print every page",
    r"full book",
    r"entire document",
    r"full private document",
    r"reveal.*full.*document",
    r"teljes könyv",
    r"minden oldal",
)


def detect_exfiltration_request(text: str) -> bool:
    """Return True when a request asks for excessive source reproduction."""

    normalized = text.lower()
    return any(re.search(pattern, normalized) for pattern in EXFILTRATION_PATTERNS)


def sanitize_quote(text: str, max_chars: int = 900) -> str:
    """Normalize whitespace and enforce the configured excerpt-length limit."""

    normalized = " ".join(text.split())
    return normalized[:max_chars]


def validate_request(question: str, config: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate user input against configured content-access guardrails."""

    reject_exfiltration = config.get("guardrails", {}).get(
        "reject_full_document_exfiltration", True
    )
    if reject_exfiltration and detect_exfiltration_request(question):
        return (
            False,
            "Request attempts to reproduce excessive source content. Ask for a summary, "
            "explanation, or a short cited excerpt instead.",
        )
    return True, None


# Backward-compatible name used by earlier documentation/tests.
EXFIL_PATTERNS = EXFILTRATION_PATTERNS

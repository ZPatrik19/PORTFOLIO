from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Iterable


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class DiagnosticFinding:
    code: str
    severity: Severity
    message: str

    def to_dict(self) -> dict[str, str]:
        payload = asdict(self)
        payload["severity"] = str(self.severity)
        return payload


def diagnose_retrieval(
    *,
    retrieved_ids: list[str],
    relevant_ids: set[str],
    source_ids: Iterable[str] = (),
    latency_ms: float | None = None,
    latency_budget_ms: float | None = None,
    top_k: int = 5,
) -> list[DiagnosticFinding]:
    findings: list[DiagnosticFinding] = []
    visible = retrieved_ids[:top_k]
    relevant_positions = [index + 1 for index, item in enumerate(visible) if item in relevant_ids]

    if relevant_ids and not relevant_positions:
        findings.append(
            DiagnosticFinding(
                "retrieval.no_hit", Severity.ERROR, "A Top-K találatok között nincs címkézett releváns chunk."
            )
        )
    elif relevant_positions and relevant_positions[0] > max(1, min(3, top_k)):
        findings.append(
            DiagnosticFinding(
                "retrieval.late_hit",
                Severity.WARNING,
                f"Az első releváns chunk csak a(z) {relevant_positions[0]}. helyen jelenik meg.",
            )
        )

    duplicate_ratio = 1.0 - len(set(visible)) / max(1, len(visible))
    if duplicate_ratio > 0.2:
        findings.append(
            DiagnosticFinding(
                "retrieval.duplicates", Severity.WARNING, f"A Top-K találatok {duplicate_ratio:.0%}-a duplikált chunk."
            )
        )

    sources = [str(source) for source in source_ids][:top_k]
    non_empty_sources = [source for source in sources if source]
    if len(non_empty_sources) >= 3:
        diversity = len(set(non_empty_sources)) / len(non_empty_sources)
        if diversity < 0.5:
            findings.append(
                DiagnosticFinding(
                    "retrieval.low_source_diversity",
                    Severity.WARNING,
                    f"Alacsony forrásdiverzitás a Top-K-ban ({diversity:.2f}).",
                )
            )

    if latency_ms is not None and latency_budget_ms is not None and latency_ms > latency_budget_ms:
        findings.append(
            DiagnosticFinding(
                "retrieval.latency_budget",
                Severity.WARNING,
                f"A retrieval {latency_ms:.1f} ms, ami meghaladja a {latency_budget_ms:.1f} ms budgetet.",
            )
        )

    if not findings:
        findings.append(
            DiagnosticFinding("retrieval.ok", Severity.INFO, "Nem észlelhető alapvető retrieval failure pattern.")
        )
    return findings


def diagnose_generation(
    *,
    citation_accuracy: float,
    citation_coverage: float,
    key_fact_coverage: float,
    context_utilization: float,
    generation_latency_ms: float | None = None,
    latency_budget_ms: float | None = None,
) -> list[DiagnosticFinding]:
    findings: list[DiagnosticFinding] = []
    if citation_accuracy < 0.95:
        findings.append(
            DiagnosticFinding(
                "generation.invalid_citation", Severity.ERROR, f"A citation accuracy csak {citation_accuracy:.0%}."
            )
        )
    if citation_coverage < 0.6:
        findings.append(
            DiagnosticFinding(
                "generation.citation_gap",
                Severity.WARNING,
                f"A válaszmondatok hivatkozási lefedettsége {citation_coverage:.0%}.",
            )
        )
    if key_fact_coverage < 0.6:
        findings.append(
            DiagnosticFinding(
                "generation.key_fact_gap", Severity.WARNING, f"A várt kulcstények lefedettsége {key_fact_coverage:.0%}."
            )
        )
    if context_utilization < 0.35:
        findings.append(
            DiagnosticFinding(
                "generation.low_context_use",
                Severity.WARNING,
                f"A lexikális context-utilization proxy alacsony ({context_utilization:.0%}).",
            )
        )
    if (
        generation_latency_ms is not None
        and latency_budget_ms is not None
        and generation_latency_ms > latency_budget_ms
    ):
        findings.append(
            DiagnosticFinding(
                "generation.latency_budget",
                Severity.WARNING,
                f"A generálás {generation_latency_ms:.1f} ms, ami meghaladja a {latency_budget_ms:.1f} ms budgetet.",
            )
        )
    if not findings:
        findings.append(
            DiagnosticFinding("generation.ok", Severity.INFO, "Nem észlelhető alapvető generation failure pattern.")
        )
    return findings


def failure_counts(findings: Iterable[DiagnosticFinding]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.code] = counts.get(finding.code, 0) + 1
    return counts

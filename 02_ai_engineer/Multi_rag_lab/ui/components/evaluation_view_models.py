from __future__ import annotations

from typing import Any, Iterable, Mapping

import pandas as pd


RETRIEVAL_HU_COLUMNS: dict[str, str] = {
    "recall_at_k": "Recall@K",
    "precision_at_k": "Precision@K",
    "f1_at_k": "F1@K",
    "hit_rate_at_k": "Hit Rate@K",
    "mrr": "MRR",
    "map_at_k": "MAP@K",
    "ndcg_at_k": "nDCG@K",
    "mean_first_relevant_rank": "Első releváns rang",
    "reciprocal_rank_at_k": "MRR@K",
    "r_precision": "R-Precision",
    "context_precision_at_k": "Context Precision@K",
    "recall_ci_low": "Recall CI alsó",
    "recall_ci_high": "Recall CI felső",
    "ndcg_ci_low": "nDCG CI alsó",
    "ndcg_ci_high": "nDCG CI felső",
    "no_hit_rate": "No-hit arány",
    "late_hit_rate": "Késői találat arány",
    "source_diversity_at_k": "Forrásdiverzitás",
    "duplicate_ratio_at_k": "Duplikációs arány",
    "mean_latency_ms": "Átlagos késleltetés ms",
    "p50_latency_ms": "P50 késleltetés ms",
    "p95_latency_ms": "P95 késleltetés ms",
    "p99_latency_ms": "P99 késleltetés ms",
    "latency_cv": "Latency CV",
    "mean_latency_ci_low_ms": "Átlag latency CI alsó ms",
    "mean_latency_ci_high_ms": "Átlag latency CI felső ms",
    "queries_per_second": "QPS",
    "labeling_coverage": "Címkézési lefedettség",
}

RAG_HU_COLUMNS: dict[str, str] = {
    "rag_strategy": "Stratégia",
    "citation_accuracy": "Hivatkozási pontosság",
    "citation_coverage": "Hivatkozási lefedettség",
    "citation_source_coverage": "Forráslefedettség",
    "citation_density_per_100_words": "Hivatkozássűrűség / 100 szó",
    "key_fact_coverage": "Kulcstény-lefedettség",
    "key_fact_ci_low": "Kulcstény CI alsó",
    "key_fact_ci_high": "Kulcstény CI felső",
    "context_utilization": "Kontextus-kihasználtság",
    "answer_redundancy": "Redundancia proxy",
    "fallback_rate": "Fallback arány",
    "repair_rate": "Repair arány",
    "mean_answer_tokens": "Átlagos választoken",
    "mean_retrieval_latency_ms": "Visszakeresés ms",
    "mean_reranking_latency_ms": "Újrarangsorolás ms",
    "mean_generation_latency_ms": "Generálás ms",
    "mean_ttft_ms": "TTFT ms",
    "mean_tokens_per_second": "Token/s",
    "mean_total_latency_ms": "Átlagos teljes idő ms",
    "p50_total_latency_ms": "P50 teljes idő ms",
    "p95_total_latency_ms": "P95 teljes idő ms",
    "p99_total_latency_ms": "P99 teljes idő ms",
    "latency_cv": "Latency CV",
    "mean_total_latency_ci_low_ms": "Teljes idő CI alsó ms",
    "mean_total_latency_ci_high_ms": "Teljes idő CI felső ms",
    "mean_context_tokens": "Átlagos kontextustoken",
}

RETRIEVAL_SCORE_COLUMNS = [
    "Recall@K",
    "MRR",
    "nDCG@K",
    "F1@K",
    "Forrásdiverzitás",
    "Címkézési lefedettség",
]

RAG_SCORE_COLUMNS = [
    "Hivatkozási pontosság",
    "Hivatkozási lefedettség",
    "Forráslefedettség",
    "Kulcstény-lefedettség",
    "Kontextus-kihasználtság",
]


def _clip01(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return numeric.clip(lower=0.0, upper=1.0)


def _scale_inverse(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return pd.Series([1.0] * len(numeric), index=numeric.index, dtype=float)
    filled = numeric.fillna(float(numeric.median()))
    lo, hi = float(filled.min()), float(filled.max())
    if hi - lo < 1e-9:
        return pd.Series([1.0] * len(filled), index=filled.index, dtype=float)
    return 1.0 - ((filled - lo) / (hi - lo))


def _records_frame(rows: Iterable[Mapping[str, Any] | Any]) -> pd.DataFrame:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, Mapping):
            normalized.append(dict(row))
        elif hasattr(row, "__dict__"):
            normalized.append(dict(vars(row)))
        else:
            raise TypeError(f"Nem támogatott benchmark sor: {type(row)!r}")
    return pd.DataFrame(normalized)


def _ensure_columns(frame: pd.DataFrame, columns: Iterable[str], *, default: float = 0.0) -> None:
    for column in columns:
        if column not in frame.columns:
            frame[column] = default


def build_retrieval_display_frame(rows: Iterable[Mapping[str, Any] | Any]) -> pd.DataFrame:
    frame = _records_frame(rows)
    if frame.empty:
        return frame

    for name in ("chunking", "retriever", "reranker"):
        if name not in frame.columns:
            frame[name] = "—"
        frame[name] = frame[name].fillna("—").astype(str)

    frame["Stratégia"] = frame["chunking"] + " · " + frame["retriever"] + " · " + frame["reranker"]
    display = frame.rename(columns=RETRIEVAL_HU_COLUMNS).copy()

    _ensure_columns(display, RETRIEVAL_SCORE_COLUMNS)
    _ensure_columns(
        display,
        [
            "Átlagos késleltetés ms",
            "P50 késleltetés ms",
            "P95 késleltetés ms",
            "P99 késleltetés ms",
            "QPS",
        ],
    )

    display["Összesített pontszám"] = (
        0.24 * _clip01(display["Recall@K"])
        + 0.20 * _clip01(display["MRR"])
        + 0.20 * _clip01(display["nDCG@K"])
        + 0.14 * _clip01(display["F1@K"])
        + 0.12 * _clip01(display["Forrásdiverzitás"])
        + 0.10 * _clip01(display["Címkézési lefedettség"])
    ).round(3)
    display["Hatékonysági pontszám"] = (
        0.75 * display["Összesített pontszám"] + 0.25 * _scale_inverse(display["Átlagos késleltetés ms"])
    ).round(3)
    return display


def build_rag_display_frame(rows: Iterable[Mapping[str, Any] | Any]) -> pd.DataFrame:
    frame = _records_frame(rows)
    if frame.empty:
        return frame

    display = frame.rename(columns=RAG_HU_COLUMNS).copy()
    _ensure_columns(display, RAG_SCORE_COLUMNS)
    _ensure_columns(
        display,
        [
            "TTFT ms",
            "Token/s",
            "Átlagos teljes idő ms",
            "P50 teljes idő ms",
            "P95 teljes idő ms",
            "P99 teljes idő ms",
            "Átlagos kontextustoken",
            "Visszakeresés ms",
            "Újrarangsorolás ms",
            "Generálás ms",
        ],
    )
    if "Stratégia" not in display.columns:
        display["Stratégia"] = "—"

    display["Összesített pontszám"] = (
        0.28 * _clip01(display["Hivatkozási pontosság"])
        + 0.20 * _clip01(display["Hivatkozási lefedettség"])
        + 0.16 * _clip01(display["Forráslefedettség"])
        + 0.24 * _clip01(display["Kulcstény-lefedettség"])
        + 0.12 * _clip01(display["Kontextus-kihasználtság"])
    ).round(3)
    display["Hatékonysági pontszám"] = (
        0.75 * display["Összesített pontszám"] + 0.25 * _scale_inverse(display["Átlagos teljes idő ms"])
    ).round(3)
    return display


def retrieval_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    ranked = frame.sort_values(["Összesített pontszám", "nDCG@K"], ascending=False)
    return {
        "best_configuration": str(ranked.iloc[0]["Stratégia"]),
        "best_overall_score": float(frame["Összesített pontszám"].max()),
        "best_efficiency_score": float(frame["Hatékonysági pontszám"].max()),
        "fastest_latency_ms": float(pd.to_numeric(frame["Átlagos késleltetés ms"], errors="coerce").min()),
    }


def rag_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    ranked = frame.sort_values(["Összesített pontszám", "Kulcstény-lefedettség"], ascending=False)
    return {
        "best_strategy": str(ranked.iloc[0]["Stratégia"]),
        "best_overall_score": float(frame["Összesített pontszám"].max()),
        "best_efficiency_score": float(frame["Hatékonysági pontszám"].max()),
        "best_ttft_ms": float(pd.to_numeric(frame["TTFT ms"], errors="coerce").min()),
    }

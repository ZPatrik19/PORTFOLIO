from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from rag_engine.platform.config import load_settings
from rag_engine.retrieval.rerank_lexical import LexicalReranker
from ui.components.common import cached_cross_encoder
from ui.components.i18n import RERANKER_HU

RERANKED_STRATEGIES = {"reranked", "dense-reranked"}


@dataclass(frozen=True)
class RerankerStatus:
    requested: str
    applied: str
    device: str
    active: bool
    reason: str


def build_selected_reranker(strategy: str, *, allow_inactive: bool = False):
    """Create the globally selected reranker consistently across pages.

    Reranking is applied only by RAG strategies that have an explicit reranking stage.
    The status object is returned even when the reranker is inactive so the UI never
    silently hides this information.
    """
    requested = str(st.session_state.get("reranker_mode", "lexical"))
    requested_device = str(st.session_state.get("reranker_device", "auto"))

    if strategy not in RERANKED_STRATEGIES and not allow_inactive:
        return None, RerankerStatus(
            requested=requested,
            applied="none",
            device="—",
            active=False,
            reason="Az aktuális RAG stratégia nem tartalmaz újrarangsorolási lépést.",
        )

    if requested == "none":
        if strategy in RERANKED_STRATEGIES and not allow_inactive:
            reranker = LexicalReranker()
            return reranker, RerankerStatus(
                requested=requested,
                applied="lexical",
                device=str(getattr(reranker, "device", "cpu")),
                active=True,
                reason="Az újrarangsorolt RAG stratégia kötelezően rerankel; a globális 'nincs' kéréshez kontrollált lexikális fallback aktív.",
            )
        return None, RerankerStatus(
            requested=requested,
            applied="none",
            device="—",
            active=False,
            reason="A globális konfigurációban az újrarangsorolás ki van kapcsolva.",
        )

    if requested == "cross-encoder":
        settings = load_settings()
        try:
            reranker = cached_cross_encoder(settings.reranker_model, requested_device)
            return reranker, RerankerStatus(
                requested=requested,
                applied="cross-encoder",
                device=str(getattr(reranker, "device", requested_device)),
                active=True,
                reason="A kiválasztott Cross-Encoder aktív.",
            )
        except Exception as exc:
            reranker = LexicalReranker()
            return reranker, RerankerStatus(
                requested=requested,
                applied="lexical",
                device="cpu",
                active=True,
                reason=f"Cross-Encoder nem volt elérhető; lexikális fallback aktív: {exc}",
            )

    reranker = LexicalReranker()
    return reranker, RerankerStatus(
        requested=requested,
        applied="lexical",
        device=str(getattr(reranker, "device", "cpu")),
        active=True,
        reason="A lexikális újrarangsoroló aktív.",
    )


def reranker_status_text(status: RerankerStatus) -> str:
    applied = RERANKER_HU.get(status.applied, status.applied)
    return f"{applied} · {status.device}" if status.active else f"{applied} · inaktív"

from __future__ import annotations

# Semantic palette: keep the same concept on the same color across the whole UI.
SEMANTIC_COLORS = {
    "data": "#4F8EF7",
    "chunking": "#35B9B0",
    "embedding": "#2BC58A",
    "retrieval": "#F2A65A",
    "reranking": "#E0B04B",
    "generation": "#9B7BF7",
    "evaluation": "#F06F8A",
    "cpu": "#55B5E8",
    "cuda": "#35C98E",
    "success": "#35C98E",
    "warning": "#F2B45E",
    "error": "#F16F7A",
    "muted": "#8FA2BE",
}

CHART_PALETTE = [
    SEMANTIC_COLORS["data"],
    SEMANTIC_COLORS["embedding"],
    SEMANTIC_COLORS["retrieval"],
    SEMANTIC_COLORS["generation"],
    SEMANTIC_COLORS["evaluation"],
    SEMANTIC_COLORS["cpu"],
    "#B7D56B",
    SEMANTIC_COLORS["reranking"],
]

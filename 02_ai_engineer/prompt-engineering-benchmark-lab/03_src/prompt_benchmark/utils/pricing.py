from __future__ import annotations

from typing import Any


def calculate_estimated_cost(
    input_tokens: int,
    output_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float,
) -> float:
    """Estimate request cost in USD from token counts and per-million-token prices."""
    return (
        input_tokens * input_price_per_million / 1_000_000
        + output_tokens * output_price_per_million / 1_000_000
    )


def get_provider_pricing(config: dict[str, Any], provider: str) -> tuple[float, float]:
    """Read provider-specific prices from configs/pricing.yaml."""
    section = config.get(provider, {})
    if not isinstance(section, dict):
        return 0.0, 0.0
    return (
        float(section.get("input_per_million_tokens", 0.0) or 0.0),
        float(section.get("output_per_million_tokens", 0.0) or 0.0),
    )

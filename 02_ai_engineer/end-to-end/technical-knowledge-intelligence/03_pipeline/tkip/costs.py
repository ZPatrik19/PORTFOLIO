from __future__ import annotations

from typing import Any


def estimate_gemini_cost_usd(
    input_tokens: int | None,
    output_tokens: int | None,
    cfg: dict[str, Any],
) -> float | None:
    """Estimate request cost from configured Gemini token pricing."""

    if input_tokens is None or output_tokens is None:
        return None

    gemini = cfg.get("gemini", {})
    input_price = gemini.get("pricing_usd_per_1m_input_tokens")
    output_price = gemini.get("pricing_usd_per_1m_output_tokens")
    if input_price is None or output_price is None:
        return None

    return (input_tokens / 1_000_000) * float(input_price) + (output_tokens / 1_000_000) * float(
        output_price
    )

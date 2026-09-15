from __future__ import annotations

def estimate_gemini_cost_usd(input_tokens:int|None,output_tokens:int|None,cfg)->float|None:
    if input_tokens is None or output_tokens is None: return None
    g=cfg.get("gemini",{})
    pin=g.get("pricing_usd_per_1m_input_tokens")
    pout=g.get("pricing_usd_per_1m_output_tokens")
    if pin is None or pout is None: return None
    return (input_tokens/1_000_000)*float(pin)+(output_tokens/1_000_000)*float(pout)

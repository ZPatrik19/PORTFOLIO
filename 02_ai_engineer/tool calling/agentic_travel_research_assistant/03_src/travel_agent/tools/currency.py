from __future__ import annotations

import json
import os
from decimal import Decimal, ROUND_HALF_UP
from urllib.request import urlopen

from pydantic import BaseModel, ConfigDict, Field, field_validator

from travel_agent.data_store import read_csv
from .base import BaseTool


class CurrencyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amount: float = Field(ge=0)
    from_currency: str = Field(min_length=3, max_length=3)
    to_currency: str = Field(min_length=3, max_length=3)

    @field_validator("from_currency", "to_currency")
    @classmethod
    def upper(cls, value: str) -> str:
        return value.upper()


class CurrencyTool(BaseTool[CurrencyInput]):
    name = "convert_currency"
    description = "Convert currency using the live Frankfurter v2 rate API; if unavailable, use the local fallback-rate dataset."
    input_model = CurrencyInput

    @staticmethod
    def _live(base: str, quote: str) -> tuple[Decimal, str]:
        with urlopen(f"https://api.frankfurter.dev/v2/rate/{base.lower()}/{quote.lower()}", timeout=4) as resp:
            payload = json.loads(resp.read().decode("utf-8"), parse_float=Decimal)
        return Decimal(str(payload["rate"])), str(payload.get("date", "latest"))

    @staticmethod
    def _fallback(base: str, quote: str) -> tuple[Decimal, str]:
        rates={r["quote"].upper(): Decimal(r["rate"]) for r in read_csv("fx_rates_fallback.csv") if r["base"].upper()=="EUR"}
        if base not in rates or quote not in rates:
            raise ValueError(f"Unsupported fallback currency pair: {base}/{quote}")
        rate = rates[quote] / rates[base]
        return rate, "2026-09-12"

    def execute(self, arguments: CurrencyInput) -> dict:
        if arguments.from_currency == arguments.to_currency:
            return {"amount": arguments.amount, "from_currency": arguments.from_currency, "to_currency": arguments.to_currency, "converted_amount": arguments.amount, "rate": 1.0, "source": "identity"}
        mode = os.getenv("TRAVEL_DATA_MODE", "auto").strip().lower()
        if mode == "local":
            rate, rate_date = self._fallback(arguments.from_currency, arguments.to_currency)
            source = "01_data/raw/fx_rates_fallback.csv"
            fallback = True
            live_error = None
        elif mode == "live":
            rate, rate_date = self._live(arguments.from_currency, arguments.to_currency)
            source = "Frankfurter v2 live API"
            fallback = False
            live_error = None
        else:
            try:
                rate, rate_date = self._live(arguments.from_currency, arguments.to_currency)
                source = "Frankfurter v2 live API"
                fallback = False
                live_error = None
            except Exception as exc:
                rate, rate_date = self._fallback(arguments.from_currency, arguments.to_currency)
                source = "01_data/raw/fx_rates_fallback.csv"
                fallback = True
                live_error = type(exc).__name__
        amount=Decimal(str(arguments.amount))
        converted=(amount*rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        out={"amount":arguments.amount,"from_currency":arguments.from_currency,"to_currency":arguments.to_currency,"converted_amount":float(converted),"rate":float(rate),"rate_date":rate_date,"source":source,"fallback":fallback}
        if fallback and live_error: out["live_api_error"]=live_error
        return out

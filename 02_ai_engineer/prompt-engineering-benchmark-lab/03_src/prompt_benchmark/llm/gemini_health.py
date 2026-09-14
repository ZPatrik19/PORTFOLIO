from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GeminiHealthResult:
    ok: bool
    model: str
    http_status: int | None
    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    error: str | None = None


def _extract_text(payload: dict[str, Any]) -> str:
    for candidate in payload.get("candidates", []) or []:
        content = candidate.get("content") or {}
        for part in content.get("parts", []) or []:
            text = part.get("text")
            if text:
                return str(text)
    return ""


def test_gemini_connection(
    api_key: str | None = None,
    model: str | None = None,
    timeout_seconds: float = 30.0,
    structured: bool = False,
) -> GeminiHealthResult:
    """Send one tiny Gemini GenerateContent request without logging the API key.

    This health check intentionally uses the REST API and Python stdlib so it can
    diagnose the key/model/network even if the google-genai SDK is not installed
    correctly yet. The normal benchmark still uses the provider adapter.
    """
    key = (api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    chosen_model = (model or os.getenv("GEMINI_MODEL") or "gemini-3.5-flash-lite").strip()
    if not key:
        return GeminiHealthResult(False, chosen_model, None, "", 0, 0, 0, "GEMINI_API_KEY is not configured.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{chosen_model}:generateContent"
    body: dict[str, Any] = {
        "system_instruction": {
            "parts": [
                {
                    "text": (
                        "You are a support-ticket classifier. Classify into exactly one of: "
                        "api, billing, cancellation, complaint, technical, upgrade."
                    )
                }
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "I was charged twice for the same invoice this month."}],
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 32,
            "thinkingConfig": {"thinkingLevel": "minimal"},
        },
    }
    if structured:
        body["generationConfig"].update(
            {
                "responseMimeType": "application/json",
                "responseJsonSchema": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "enum": ["api", "billing", "cancellation", "complaint", "technical", "upgrade"],
                        }
                    },
                    "required": ["label"],
                    "additionalProperties": False,
                },
            }
        )

    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - fixed HTTPS endpoint
            raw = response.read().decode("utf-8")
            payload = json.loads(raw)
            usage = payload.get("usageMetadata") or {}
            return GeminiHealthResult(
                ok=200 <= int(response.status) < 300,
                model=chosen_model,
                http_status=int(response.status),
                text=_extract_text(payload).strip(),
                input_tokens=int(usage.get("promptTokenCount") or 0),
                output_tokens=int(usage.get("candidatesTokenCount") or 0),
                total_tokens=int(usage.get("totalTokenCount") or 0),
                error=None,
            )
    except urllib.error.HTTPError as exc:
        try:
            raw_error = exc.read().decode("utf-8", errors="replace")
            data = json.loads(raw_error)
            message = ((data.get("error") or {}).get("message") or raw_error)[:800]
        except (json.JSONDecodeError, UnicodeDecodeError, OSError, AttributeError, TypeError):
            message = str(exc)
        return GeminiHealthResult(False, chosen_model, int(exc.code), "", 0, 0, 0, f"Gemini API HTTP {exc.code}: {message}")
    except urllib.error.URLError as exc:
        return GeminiHealthResult(False, chosen_model, None, "", 0, 0, 0, f"Network/DNS error: {exc.reason}")
    except Exception as exc:
        return GeminiHealthResult(False, chosen_model, None, "", 0, 0, 0, f"{type(exc).__name__}: {exc}")

from __future__ import annotations

from tkip.config import gemini_api_key, load_config
from tkip.gemini_service import GeminiService


def main() -> int:
    """Validate Gemini connectivity when a key is configured.

    Missing credentials are not considered a setup failure because retrieval and
    deterministic offline evaluation remain available without Gemini.
    """
    key = gemini_api_key()
    if not key:
        print(
            "GEMINI_API_KEY not configured. Retrieval/local demo mode remains available; "
            "generated grounded answers require Gemini."
        )
        return 0

    service = GeminiService(load_config(), api_key=key)
    ok, message = service.test_connection()
    if ok:
        print(f"Gemini API reachable: {message}")
        return 0

    print(f"Gemini API check failed: {message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

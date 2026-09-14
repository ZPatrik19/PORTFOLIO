"""EN: Transient-error retry classification and fail-fast authentication behavior.

HU: Az átmeneti hibák retry-besorolását és az autentikációs hibák fail-fast kezelését ellenőrzi.
"""

from __future__ import annotations

from prompt_benchmark.llm.base import is_retryable_provider_error


def test_retryable_provider_errors() -> None:
    """EN: Checks that transient timeout/rate-limit/server errors are classified as retryable.

    HU: Ellenőrzi, hogy timeout/rate-limit/server jellegű átmeneti hibák retryable kategóriába kerülnek.
    """
    assert is_retryable_provider_error(TimeoutError("request timeout"))
    assert is_retryable_provider_error(RuntimeError("429 rate limit"))
    assert is_retryable_provider_error(RuntimeError("503 service unavailable"))


def test_authentication_errors_are_not_retried() -> None:
    """EN: Ensures invalid credentials fail immediately instead of wasting retries and quota.

    HU: Biztosítja, hogy hibás credential esetén ne pazaroljon retry-t és quotát a rendszer.
    """
    assert not is_retryable_provider_error(RuntimeError("401 unauthorized invalid api key"))
    assert not is_retryable_provider_error(RuntimeError("400 bad request"))

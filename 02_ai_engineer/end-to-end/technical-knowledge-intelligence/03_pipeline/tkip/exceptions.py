"""Domain-specific exception hierarchy for the knowledge platform."""

from __future__ import annotations


class TKIPError(Exception):
    """Base exception for expected platform failures."""


class ConfigurationError(TKIPError):
    """Raised when application configuration is missing or invalid."""


class DependencyError(TKIPError):
    """Raised when an optional or required runtime dependency is unavailable."""


class IndexNotReadyError(TKIPError):
    """Raised when a requested retrieval index cannot be loaded or built."""


class DocumentParsingError(TKIPError):
    """Raised when a document cannot be parsed into structured blocks."""


class ExternalServiceError(TKIPError):
    """Base class for external API/service failures."""


class AuthenticationError(ExternalServiceError):
    """Raised when an external service rejects credentials."""


class QuotaExceededError(ExternalServiceError):
    """Raised when an external AI provider quota or rate limit is exhausted."""


class ExternalServiceTimeoutError(ExternalServiceError):
    """Raised when an external service call exceeds the configured timeout."""


class StructuredOutputError(ExternalServiceError):
    """Raised when a model response cannot be validated against the response schema."""

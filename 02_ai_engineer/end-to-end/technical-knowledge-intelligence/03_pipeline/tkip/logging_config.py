"""Central logging configuration used by CLI, API and batch workflows."""

from __future__ import annotations

import logging
import logging.config
from typing import Any


def configure_logging(config: dict[str, Any] | None = None) -> None:
    """Configure concise production-style logging.

    The application intentionally logs to stdout by default because Docker and
    Kubernetes collect process output. Structured request telemetry remains a
    separate concern handled by :mod:`tkip.monitoring`.
    """

    level = str((config or {}).get("logging", {}).get("level", "INFO")).upper()
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "level": level,
                }
            },
            "root": {"handlers": ["console"], "level": level},
        }
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module logger without introducing global side effects."""

    return logging.getLogger(name)

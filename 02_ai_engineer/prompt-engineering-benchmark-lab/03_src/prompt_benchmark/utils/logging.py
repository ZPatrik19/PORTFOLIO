"""Central logging configuration for CLI, UI and batch workflows."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from prompt_benchmark.paths import PATHS

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(
    level: int | str = logging.INFO,
    *,
    log_file: str | Path | None = None,
    console: bool = True,
) -> None:
    """Configure application logging once, with optional rotating file output."""
    root = logging.getLogger()
    root.setLevel(level)
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

    # Idempotent setup: avoid duplicated handlers on Streamlit reruns.
    if console and not any(getattr(h, "_prompt_benchmark_console", False) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler._prompt_benchmark_console = True  # type: ignore[attr-defined]
        root.addHandler(handler)

    target = Path(log_file) if log_file else PATHS.logs / "application.log"
    target.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(target.resolve())
    if not any(getattr(h, "baseFilename", None) == resolved for h in root.handlers):
        file_handler = RotatingFileHandler(target, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

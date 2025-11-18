"""
Structured logging for V2D.

Provides consistent logging across all components with support for
job context and structured data.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class V2DLogger(logging.Logger):
    """Custom logger with V2D-specific features."""

    def __init__(self, name: str, level: int = logging.NOTSET):
        super().__init__(name, level)
        self._job_id: str | None = None

    def set_job_context(self, job_id: str | None) -> None:
        """Set the current job context for log messages."""
        self._job_id = job_id

    def _log_with_context(
        self,
        level: int,
        msg: str,
        args: tuple,
        exc_info: Any = None,
        extra: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Log with job context."""
        if extra is None:
            extra = {}

        if self._job_id:
            extra["job_id"] = self._job_id

        super()._log(level, msg, args, exc_info=exc_info, extra=extra, **kwargs)


class V2DFormatter(logging.Formatter):
    """Custom formatter for V2D logs."""

    def __init__(self, include_timestamp: bool = True):
        self.include_timestamp = include_timestamp
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record."""
        parts = []

        # Timestamp
        if self.include_timestamp:
            timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
            parts.append(f"[{timestamp}]")

        # Level
        parts.append(f"[{record.levelname}]")

        # Job ID if present
        job_id = getattr(record, "job_id", None)
        if job_id:
            parts.append(f"[{job_id}]")

        # Logger name
        parts.append(f"[{record.name}]")

        # Message
        parts.append(record.getMessage())

        return " ".join(parts)


# Module-level logger storage
_loggers: dict[str, logging.Logger] = {}
_initialized = False


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    console: bool = True,
) -> None:
    """
    Setup logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        console: Whether to output to console
    """
    global _initialized

    # Set logging class
    logging.setLoggerClass(V2DLogger)

    # Get root logger for v2d
    root_logger = logging.getLogger("v2d")
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(V2DFormatter(include_timestamp=True))
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(V2DFormatter(include_timestamp=True))
        root_logger.addHandler(file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance
    """
    global _initialized

    # Ensure logging is set up
    if not _initialized:
        setup_logging()

    # Create or get logger
    if name not in _loggers:
        # Ensure name is under v2d namespace
        if not name.startswith("v2d"):
            name = f"v2d.{name}"

        logger = logging.getLogger(name)
        _loggers[name] = logger

    return _loggers[name]


def set_job_context(job_id: str | None) -> None:
    """
    Set job context for all loggers.

    Args:
        job_id: Job ID or None to clear
    """
    for logger in _loggers.values():
        if isinstance(logger, V2DLogger):
            logger.set_job_context(job_id)

"""Centralised logging configuration for the application."""

import logging
import sys

from pydantic import ValidationError

from app.core.config import get_settings

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _resolve_level(level: str | None) -> str:
    if level:
        return level.upper()
    try:
        return get_settings().LOG_LEVEL
    except ValidationError:
        # Logging must stay usable even when the environment is misconfigured,
        # otherwise the config error itself cannot be reported.
        return "INFO"


def setup_logging(level: str | None = None, *, force: bool = False) -> None:
    """Attach a formatted console handler to the root logger.

    Repeated calls are ignored unless ``force`` is set, which keeps duplicate
    handlers (and duplicate log lines) from piling up on module reloads.
    """
    global _configured
    if _configured and not force:
        return

    root = logging.getLogger()
    if force:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            handler.close()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))

    root.addHandler(handler)
    root.setLevel(_resolve_level(level))

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger, configuring the root handler on first use."""
    setup_logging()
    return logging.getLogger(name)

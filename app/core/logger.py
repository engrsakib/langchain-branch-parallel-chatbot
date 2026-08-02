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
        return "INFO"


def setup_logging(level: str | None = None, *, force: bool = False) -> None:
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
    setup_logging()
    return logging.getLogger(name)

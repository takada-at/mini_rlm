import logging
import os
from pathlib import Path

_LOGGER_NAME = "mini_rlm"
_DEFAULT_LOG_PATH = "mini_rlm_debug.log"
_initialized_log_path: Path | None = None


def _resolve_log_path() -> Path:
    if _initialized_log_path is not None:
        return _initialized_log_path

    configured = os.getenv("MINI_RLM_LOG_FILE")
    if configured is None or configured.strip() == "":
        return Path(_DEFAULT_LOG_PATH)
    return Path(configured)


def initialize_log_path(log_path: str | Path | None = None) -> Path:
    global _initialized_log_path

    if log_path is None:
        resolved_log_path = _resolve_log_path()
    else:
        resolved_log_path = Path(log_path)

    _initialized_log_path = resolved_log_path
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
    return resolved_log_path


def get_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger

    try:
        log_path = initialize_log_path()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s"
        )
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
    except OSError:
        # Debug logging must not block the REPL session from starting.
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        return logger

    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


def get_log_file_path() -> Path:
    return _resolve_log_path()

import logging
import os
from pathlib import Path

_LOGGER_NAME = "mini_rlm"
_DEFAULT_LOG_PATH = "mini_rlm_debug.log"


def _resolve_log_path() -> Path:
    configured = os.getenv("MINI_RLM_LOG_FILE")
    if configured is None or configured.strip() == "":
        return Path(_DEFAULT_LOG_PATH)
    return Path(configured)


def get_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger

    log_path = _resolve_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


def get_log_file_path() -> Path:
    return _resolve_log_path()

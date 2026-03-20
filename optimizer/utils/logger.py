"""
Logger
======
Centralized logging setup for the OpenCL AI Optimizer.
Outputs coloured, structured log lines to console + optional file.

Usage:
    from optimizer.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Device ready.")
"""

import logging
import sys
from typing import Optional

# ANSI colour codes
_RESET  = "\033[0m"
_COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
}


class _ColourFormatter(logging.Formatter):
    """Formatter that prepends ANSI colour codes to the level name."""

    FMT = "[%(name)s] %(levelname)s — %(message)s"

    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelname, _RESET)
        record.levelname = f"{colour}{record.levelname}{_RESET}"
        return logging.Formatter(self.FMT).format(record)


# Root logger for the entire package
_PKG_LOGGER_NAME = "opencl_optimizer"
_root_logger: Optional[logging.Logger] = None


def _init_root_logger(level: int = logging.INFO) -> logging.Logger:
    """Build and cache the package root logger."""
    global _root_logger
    if _root_logger is not None:
        return _root_logger

    logger = logging.getLogger(_PKG_LOGGER_NAME)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_ColourFormatter())
        logger.addHandler(handler)

    logger.propagate = False
    _root_logger = logger
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger under the package root.

    Args:
        name : Usually __name__ of the calling module.

    Returns:
        A configured Logger instance.
    """
    _init_root_logger()
    # Strip the package prefix for clean output
    short = name.replace("optimizer.", "")
    return logging.getLogger(f"{_PKG_LOGGER_NAME}.{short}")


def set_log_level(level: str) -> None:
    """
    Change the package-wide log verbosity.

    Args:
        level : 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'
    """
    numeric = getattr(logging, level.upper(), logging.INFO)
    _init_root_logger()
    logging.getLogger(_PKG_LOGGER_NAME).setLevel(numeric)
    for handler in logging.getLogger(_PKG_LOGGER_NAME).handlers:
        handler.setLevel(numeric)

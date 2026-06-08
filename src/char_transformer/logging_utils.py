"""Minimal logging setup shared across scripts and modules."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module logger with a single stdout handler.

    Configures the root handler once so repeated calls across modules do not
    attach duplicate handlers (which would print every line multiple times).
    """
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(level)
        _CONFIGURED = True

    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

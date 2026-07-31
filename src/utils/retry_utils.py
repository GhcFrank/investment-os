"""
Small bounded retry helper for external data providers.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from typing import TypeVar


ResultT = TypeVar("ResultT")


def short_error(error: Exception | str, limit: int = 180) -> str:
    """
    Remove URLs and collapse an exception to a log-safe summary.
    """

    text = re.sub(r"https?://\S+", "[url removed]", str(error))
    text = re.sub(r"\s+", " ", text).strip()
    return (text or "unknown error")[:limit]


def retry_call(
    operation: Callable[[], ResultT],
    *,
    label: str,
    max_attempts: int = 3,
    sleep_func: Callable[[float], None] = time.sleep,
    logger: logging.Logger | None = None,
) -> ResultT:
    """
    Run an operation with bounded exponential backoff.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as error:
            if attempt == max_attempts:
                raise
            delay = float(2 ** (attempt - 1))
            if logger is not None:
                logger.warning(
                    "%s failed on attempt %s/%s; retrying in %.0fs: %s",
                    label,
                    attempt,
                    max_attempts,
                    delay,
                    short_error(error),
                )
            sleep_func(delay)

    raise RuntimeError("Retry loop ended unexpectedly")

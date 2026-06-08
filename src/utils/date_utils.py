"""
Shared date/time helpers.
"""

from datetime import datetime
from zoneinfo import ZoneInfo


MARKET_TIMEZONE = ZoneInfo("America/New_York")


def today_et_str() -> str:
    """
    Return today's date in America/New_York.
    """

    return datetime.now(MARKET_TIMEZONE).strftime("%Y-%m-%d")


def now_et_str() -> str:
    """
    Return current timestamp in America/New_York.
    """

    return datetime.now(MARKET_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S %Z")

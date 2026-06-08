"""
Shared helpers for Polymarket earnings prediction scripts.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parents[2]
POLYMARKET_GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
MARKET_TIMEZONE = ZoneInfo("America/New_York")

USER_AGENT = (
    "Investment OS/1.0 "
    "(https://github.com; personal research tool)"
)


def today_et() -> str:
    """
    Return today's date in America/New_York.
    """

    return datetime.now(MARKET_TIMEZONE).date().isoformat()


def now_et() -> str:
    """
    Return the current timestamp in America/New_York.
    """

    return datetime.now(MARKET_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S %Z")


def fetch_gamma_json(
    path: str,
    params: dict | None = None,
) -> list | dict:
    """
    Fetch JSON from Polymarket Gamma API.
    """

    response = requests.get(
        f"{POLYMARKET_GAMMA_BASE_URL}{path}",
        params=params,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
        timeout=60,
    )

    response.raise_for_status()

    return response.json()


def parse_jsonish_list(value) -> list:
    """
    Parse Polymarket fields that may arrive as a list or as a JSON string.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]

    if isinstance(parsed, list):
        return parsed

    return [parsed]


def as_json_string(value) -> str:
    """
    Store a value as compact JSON for CSV output.
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
    )

def to_float(value):
    """
    把输入值安全转换成 float。

    这个函数要处理几类情况：

    1. None
    2. 空字符串
    3. pandas 的 NA / NaN
    4. 普通数字
    5. 字符串数字，例如 "0.57"

    如果无法转换，就返回 None。
    """

    if value is None:
        return None

    # pandas 的 pd.NA / NaN 需要单独处理。
    # 不能直接写 value == ""，因为 pd.NA 做布尔判断会报错。
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if value == "":
        return None

    try:
        return float(value)
    except Exception:
        return None


def normalize_bool(value) -> bool:
    """
    Convert CSV/API booleans to Python bool.
    """

    if isinstance(value, bool):
        return value

    if value is None:
        return False

    text = str(value).strip().lower()

    return text in {"true", "1", "yes", "y"}


def build_polymarket_url(slug: str) -> str:
    """
    Build a public Polymarket event URL from a slug.
    """

    if not slug:
        return ""

    return f"https://polymarket.com/event/{slug}"

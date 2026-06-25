"""
Shared helpers for News Discovery V1.
"""

from __future__ import annotations

import html
import os
import re
import time
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import requests


NEWS_USER_AGENT = os.getenv(
    "NEWS_USER_AGENT",
    "InvestmentOS-NewsBot/1.0",
)

REQUEST_RETRY_STATUSES = {429, 500, 502, 503, 504}
SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": NEWS_USER_AGENT,
        "Accept": "application/json",
    }
)

UTM_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
}


class _TextExtractor(HTMLParser):
    """
    Small standard-library HTML text extractor.
    """

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def get_text(self) -> str:
        return " ".join(self.parts)


def now_utc_iso() -> str:
    """
    Return the current UTC timestamp as an ISO 8601 string ending in Z.
    """

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def utc_today_yyyy_mm_dd() -> str:
    """
    Return today's UTC date as YYYY-MM-DD.
    """

    return datetime.now(UTC).date().isoformat()


def parse_utc_datetime(value: str) -> datetime:
    """
    Parse an ISO datetime string as UTC.
    """

    normalized = value.strip()

    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    parsed = datetime.fromisoformat(normalized)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)

    return parsed.astimezone(UTC)


def to_yyyy_mm_dd(value: object) -> str:
    """
    Convert a date/datetime-like value to YYYY-MM-DD.
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(UTC)
        return value.date().isoformat()

    if isinstance(value, date):
        return value.isoformat()

    text = str(value).strip()

    if not text:
        return ""

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text

    try:
        return parse_utc_datetime(text).date().isoformat()
    except (TypeError, ValueError):
        return ""


def format_utc(value: datetime) -> str:
    """
    Format a datetime as seconds-precision UTC ISO 8601.
    """

    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
    )


def normalize_wp_gmt_datetime(value: object) -> str:
    """
    Normalize a WordPress GMT datetime value to UTC ISO 8601.
    """

    text = str(value or "").strip()

    if not text:
        return ""

    if text.endswith("Z") or "+" in text[10:] or "-" in text[10:]:
        return format_utc(parse_utc_datetime(text))

    return format_utc(datetime.fromisoformat(text).replace(tzinfo=UTC))


def _retry_after_seconds(value: str | None) -> float | None:
    """
    Parse Retry-After as seconds or HTTP-date.
    """

    if not value:
        return None

    stripped = value.strip()

    try:
        seconds = float(stripped)
        return max(seconds, 0)
    except ValueError:
        pass

    try:
        retry_time = parsedate_to_datetime(stripped)

        if retry_time.tzinfo is None:
            retry_time = retry_time.replace(tzinfo=UTC)

        delta = retry_time.astimezone(UTC) - datetime.now(UTC)
        return max(delta.total_seconds(), 0)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


def get_json(
    url: str,
    params: dict[str, object] | None = None,
) -> requests.Response:
    """
    Fetch JSON over HTTP with targeted retries and status validation.
    """

    retry_delays = [1, 2, 4]

    for attempt_number in range(len(retry_delays) + 1):
        try:
            response = SESSION.get(url, params=params, timeout=30)

        except (requests.ConnectionError, requests.Timeout):
            if attempt_number < len(retry_delays):
                time.sleep(retry_delays[attempt_number])
                continue
            raise

        if response.status_code in REQUEST_RETRY_STATUSES:
            if attempt_number < len(retry_delays):
                retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
                time.sleep(
                    retry_after
                    if retry_after is not None
                    else retry_delays[attempt_number]
                )
                continue

            response.raise_for_status()

        response.raise_for_status()
        return response

    raise RuntimeError(f"HTTP request failed without an error: {url}")


def clean_html_text(value: object) -> str:
    """
    Convert WordPress HTML-ish values into normalized plain text.
    """

    if value is None:
        return ""

    if isinstance(value, dict):
        value = value.get("rendered", "")

    parser = _TextExtractor()
    parser.feed(str(value))
    parser.close()

    text = html.unescape(parser.get_text())
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_match_text(value: str) -> str:
    """
    Normalize text and keywords into the same matching form.
    """

    text = clean_html_text(value).lower()
    text = re.sub(r"[-/–—]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def contains_normalized_phrase(
    normalized_text: str,
    normalized_keyword: str,
) -> bool:
    """
    Match a normalized keyword inside normalized text with full boundaries.
    """

    if not normalized_text or not normalized_keyword:
        return False

    pattern = rf"(?<![a-z0-9]){re.escape(normalized_keyword)}(?![a-z0-9])"

    return re.search(pattern, normalized_text) is not None


def contains_phrase(text: str, keyword: str) -> bool:
    """
    Match a normalized keyword with complete alphanumeric boundaries.
    """

    return contains_normalized_phrase(
        normalize_match_text(text),
        normalize_match_text(keyword),
    )


def normalize_url(url: str) -> str:
    """
    Remove common tracking parameters from a URL.
    """

    if not url:
        return ""

    parts = urlsplit(url)
    kept_query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key not in UTM_PARAMS
    ]

    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(kept_query, doseq=True),
            parts.fragment,
        )
    )


def repo_relative_path(path: Path, base_dir: Path) -> str:
    """
    Convert a path to a repository-relative path when possible.
    """

    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        print(f"Warning: path is outside repository: {path}")
        return str(path)


def read_csv_safe(path: Path, columns: list[str]) -> pd.DataFrame:
    """
    Read a CSV while tolerating missing, empty, or partial files.
    """

    if not path.exists():
        return pd.DataFrame(columns=columns)

    try:
        df = pd.read_csv(path, dtype=str)

    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)

    for column in columns:
        if column not in df.columns:
            df[column] = ""

    return df.reindex(columns=columns)


def atomic_write_csv(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    """
    Write a CSV with a stable schema via same-directory atomic replace.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    output = df.copy()

    for column in columns:
        if column not in output.columns:
            output[column] = ""

    output = output.reindex(columns=columns)
    temp_path = path.with_name(f".{path.name}.tmp")
    output.to_csv(temp_path, index=False, encoding="utf-8")
    temp_path.replace(path)


def join_values(values: list[object] | set[object]) -> str:
    """
    Join non-empty values in stable alphabetical order.
    """

    cleaned = {
        str(value).strip()
        for value in values
        if value is not None and str(value).strip()
    }

    return "|".join(sorted(cleaned))


def csv_bool(value: object) -> bool:
    """
    Interpret common CSV truthy values.
    """

    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_int(value: object, default: int = 0) -> int:
    """
    Convert a value to int with a fallback.
    """

    try:
        if pd.isna(value):
            return default
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default

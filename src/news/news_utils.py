"""
Shared helpers for News Discovery V1.
"""

from __future__ import annotations

import html
import os
import re
import time
from datetime import UTC, datetime
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


def get_json(
    url: str,
    params: dict[str, object] | None = None,
) -> requests.Response:
    """
    Fetch JSON over HTTP with retries and status validation.
    """

    last_error: Exception | None = None
    retry_delays = [1, 2, 4]

    for attempt_number in range(len(retry_delays) + 1):
        try:
            response = SESSION.get(url, params=params, timeout=30)

            if response.status_code in REQUEST_RETRY_STATUSES:
                response.raise_for_status()

            response.raise_for_status()
            return response

        except requests.RequestException as error:
            last_error = error

            if attempt_number < len(retry_delays):
                time.sleep(retry_delays[attempt_number])
                continue

            raise

    if last_error is not None:
        raise last_error

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


def contains_phrase(text: str, keyword: str) -> bool:
    """
    Match a normalized keyword with complete alphanumeric boundaries.
    """

    normalized_text = normalize_match_text(text)
    normalized_keyword = normalize_match_text(keyword)

    if not normalized_text or not normalized_keyword:
        return False

    pattern = rf"(?<![a-z0-9]){re.escape(normalized_keyword)}(?![a-z0-9])"

    return re.search(pattern, normalized_text) is not None


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

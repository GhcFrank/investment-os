"""
Email summary helpers for market sentiment indicators.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
SENTIMENT_CURRENT_FILE = BASE_DIR / "data" / "market_data" / "sentiment_indicators.csv"

SENTIMENT_COLUMNS = [
    "date",
    "indicator",
    "value",
    "level",
    "source",
    "status",
    "error_message",
    "change_1d",
    "change_5d",
    "change_20d",
    "updated_at",
]


def _clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    return str(value).strip()


def _format_value(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""

    if pd.isna(numeric):
        return ""

    return f"{numeric:.2f}"


def _format_change(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""

    if pd.isna(numeric):
        return ""

    sign = "+" if numeric > 0 else ""
    return f"{sign}{numeric:.2f}"


def _indicator_label(indicator: str) -> str:
    labels = {
        "VIX": "VIX",
        "CNN_FEAR_GREED": "CNN Fear & Greed",
    }
    return labels.get(indicator, indicator)


def _format_indicator(row: pd.Series) -> list[str]:
    label = _indicator_label(_clean(row.get("indicator")))
    status = _clean(row.get("status"))

    if status != "ok":
        lines = [
            f"{label}: unavailable",
            f"- status: {status or 'unknown'}",
        ]
        error = _clean(row.get("error_message"))
        if error:
            lines.append(f"- error: {error}")
        return lines

    value = _format_value(row.get("value"))
    level = _clean(row.get("level")) or "unknown"
    lines = [f"{label}: {value} ({level})"]

    for column, label_text in [
        ("change_1d", "1d change"),
        ("change_5d", "5d change"),
        ("change_20d", "20d change"),
    ]:
        change = _format_change(row.get(column))
        if change:
            lines.append(f"- {label_text}: {change}")

    return lines


def build_sentiment_email_section(
    current_file: Path = SENTIMENT_CURRENT_FILE,
) -> str:
    if not current_file.exists():
        return "Market Sentiment\n\nNo sentiment data available."

    try:
        current = pd.read_csv(current_file, dtype=str)
    except pd.errors.EmptyDataError:
        return "Market Sentiment\n\nNo sentiment data available."

    if current.empty:
        return "Market Sentiment\n\nNo sentiment data available."

    for column in SENTIMENT_COLUMNS:
        if column not in current.columns:
            current[column] = ""

    lines = ["Market Sentiment"]
    by_indicator = {
        _clean(row.get("indicator")): row for _, row in current.iterrows()
    }

    for indicator in ["VIX", "CNN_FEAR_GREED"]:
        row = by_indicator.get(indicator)
        if row is None:
            lines.extend(["", f"{_indicator_label(indicator)}: unavailable"])
            continue

        lines.extend(["", *_format_indicator(row)])

    return "\n".join(lines)

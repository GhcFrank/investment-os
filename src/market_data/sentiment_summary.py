"""
Email summary helpers for market sentiment indicators.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
VIX_CURRENT_FILE = BASE_DIR / "data" / "market_data" / "vix.csv"
CNN_CURRENT_FILE = BASE_DIR / "data" / "market_data" / "cnn_fear_greed.csv"

VIX_COLUMNS = [
    "date",
    "vix",
    "level",
    "source",
    "status",
    "error_message",
    "change_1d",
    "change_5d",
    "change_20d",
    "updated_at",
]

CNN_COLUMNS = [
    "date",
    "fear_greed_index",
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


def _read_one_row(path: Path, columns: list[str]) -> pd.Series | None:
    if not path.exists():
        return None

    try:
        rows = pd.read_csv(path, dtype=str)
    except pd.errors.EmptyDataError:
        return None

    if rows.empty:
        return None

    for column in columns:
        if column not in rows.columns:
            rows[column] = ""

    return rows.reindex(columns=columns).tail(1).iloc[0]


def _format_indicator(
    *,
    label: str,
    row: pd.Series | None,
    value_column: str,
) -> list[str]:
    if row is None:
        return [f"{label}: unavailable"]

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

    value = _format_value(row.get(value_column))
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
    vix_file: Path = VIX_CURRENT_FILE,
    cnn_file: Path = CNN_CURRENT_FILE,
) -> str:
    lines = ["Market Sentiment"]
    lines.extend(
        [
            "",
            *_format_indicator(
                label="VIX",
                row=_read_one_row(vix_file, VIX_COLUMNS),
                value_column="vix",
            ),
            "",
            *_format_indicator(
                label="CNN Fear & Greed",
                row=_read_one_row(cnn_file, CNN_COLUMNS),
                value_column="fear_greed_index",
            ),
        ]
    )
    return "\n".join(lines)

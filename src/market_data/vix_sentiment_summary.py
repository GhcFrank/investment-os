"""Render the formal VIX signal for the unified daily email."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

import pandas as pd

from market_data.vix_sentiment_config import (
    VIX_MARKET_SENTIMENT_SIGNAL_COLUMNS,
    VIX_MARKET_SENTIMENT_SIGNAL_FILE,
)


EMAIL_TITLE = "VIX MARKET SENTIMENT"


@dataclass(frozen=True)
class VIXMarketSentimentEmailSection:
    plain_text: str
    html: str
    available: bool
    status: str


def _clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def _number(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(parsed) else parsed


def _format_level(value: object) -> str:
    parsed = _number(value)
    return "N/A" if parsed is None else f"{parsed:.2f}"


def _format_change(value: object) -> str:
    parsed = _number(value)
    if parsed is None:
        return "Insufficient history"
    return f"{parsed:+.2f} points"


def build_vix_market_sentiment_unavailable_email_section(
    status: str = "DATA_UNAVAILABLE",
) -> VIXMarketSentimentEmailSection:
    safe_status = (
        status
        if status
        in {
            "SUCCESS",
            "SUCCESS_WITH_WARNINGS",
            "INSUFFICIENT_HISTORY",
            "DATA_UNAVAILABLE",
        }
        else "DATA_UNAVAILABLE"
    )
    plain = "\n".join(
        [
            EMAIL_TITLE,
            "",
            "VIX market sentiment data unavailable",
            f"Status: {safe_status}",
        ]
    )
    html = (
        f"<section><h2>{escape(EMAIL_TITLE)}</h2>"
        "<p><strong>VIX market sentiment data unavailable</strong><br>"
        f"Status: {escape(safe_status)}</p></section>"
    )
    return VIXMarketSentimentEmailSection(
        plain_text=plain,
        html=html,
        available=False,
        status=safe_status,
    )


def _read_signal(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        rows = pd.read_csv(path, dtype=str)
    except pd.errors.EmptyDataError as error:
        raise ValueError("VIX signal file is empty") from error
    if rows.empty or list(rows.columns) != VIX_MARKET_SENTIMENT_SIGNAL_COLUMNS:
        raise ValueError("VIX signal file has an invalid schema")
    return rows.tail(1).iloc[0]


def build_vix_market_sentiment_email_section(
    signal_file: Path | str = VIX_MARKET_SENTIMENT_SIGNAL_FILE,
) -> VIXMarketSentimentEmailSection:
    """Read only the formal VIX signal and render plain/HTML alternatives."""

    try:
        row = _read_signal(Path(signal_file))
    except (OSError, ValueError, pd.errors.ParserError):
        return build_vix_market_sentiment_unavailable_email_section()

    status = _clean(row.get("data_status")).upper()
    if status == "DATA_UNAVAILABLE" or not status:
        return build_vix_market_sentiment_unavailable_email_section()

    stale = _clean(row.get("stale")).lower() in {"true", "1", "yes"}
    fields = [
        ("VIX level", _format_level(row.get("vix"))),
        ("Daily change", _format_change(row.get("change_1d"))),
        ("5-day change", _format_change(row.get("change_5d"))),
        ("20-day change", _format_change(row.get("change_20d"))),
        ("Current sentiment regime", _clean(row.get("sentiment_regime")) or "UNKNOWN"),
        ("Signal / interpretation", _clean(row.get("signal")) or "NEUTRAL"),
        ("Data date", _clean(row.get("date")) or "N/A"),
        ("Data status", status),
        ("Stale", "Yes" if stale else "No"),
    ]
    plain = "\n".join(
        [EMAIL_TITLE, "", *(f"{label}: {value}" for label, value in fields)]
    )
    rows_html = "".join(
        f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>"
        for label, value in fields
    )
    html = (
        f"<section><h2>{escape(EMAIL_TITLE)}</h2>"
        f"<table><tbody>{rows_html}</tbody></table>"
        "<p>Changes are VIX index-point changes over existing daily market "
        "observations.</p></section>"
    )
    return VIXMarketSentimentEmailSection(
        plain_text=plain,
        html=html,
        available=True,
        status=status,
    )

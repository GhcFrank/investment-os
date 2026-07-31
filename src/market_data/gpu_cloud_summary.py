"""Render the formal Vast.ai daily signal file for the unified email."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

import pandas as pd

from market_data.gpu_cloud_config import (
    GPU_CLOUD_SIGNAL_COLUMNS,
    GPU_CLOUD_SIGNALS_FILE,
    SUPPLY_SIGNAL_SCOPE,
    TRACKED_GPU_MODELS,
)
from market_data.gpu_cloud_status import (
    API_KEY_MISSING,
    PROVIDER_ERROR,
    SCHEMA_ERROR,
)


EMAIL_TITLE = "GPU CLOUD SUPPLY — VAST.AI"
EMAIL_SCOPE = "Vast.ai visible marketplace only"
UNAVAILABLE_STATUSES = {API_KEY_MISSING, PROVIDER_ERROR, SCHEMA_ERROR}


@dataclass(frozen=True)
class GPUCloudEmailSection:
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


def _format_percent(value: object) -> str:
    parsed = _number(value)
    if parsed is None:
        return "Insufficient history"
    return f"{parsed:+.1%}"


def _format_count(value: object) -> str:
    parsed = _number(value)
    if parsed is None:
        return "N/A"
    return str(int(parsed)) if parsed.is_integer() else f"{parsed:.1f}"


def _format_snapshot(value: object) -> str:
    text = _clean(value)
    if not text:
        return "N/A"
    timestamp = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return "N/A"
    return timestamp.strftime("%Y-%m-%d %H:%M UTC")


def _safe_status(status: object) -> str:
    normalized = _clean(status).upper()
    if normalized in UNAVAILABLE_STATUSES:
        return normalized
    return normalized if normalized else PROVIDER_ERROR


def build_gpu_cloud_unavailable_email_section(
    status: str = PROVIDER_ERROR,
) -> GPUCloudEmailSection:
    """Return a secret-free failure section without implying zero supply."""

    safe_status = _safe_status(status)
    plain = "\n".join(
        [
            EMAIL_TITLE,
            "",
            "GPU market data unavailable",
            f"Status: {safe_status}",
            f"Scope: {EMAIL_SCOPE}",
        ]
    )
    html = (
        f"<section><h2>{escape(EMAIL_TITLE)}</h2>"
        "<p><strong>GPU market data unavailable</strong><br>"
        f"Status: {escape(safe_status)}<br>"
        f"Scope: {escape(EMAIL_SCOPE)}</p></section>"
    )
    return GPUCloudEmailSection(
        plain_text=plain,
        html=html,
        available=False,
        status=safe_status,
    )


def _read_latest_signals(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        rows = pd.read_csv(path, dtype=str)
    except pd.errors.EmptyDataError as error:
        raise ValueError("GPU signal file is empty") from error
    if rows.empty or list(rows.columns) != GPU_CLOUD_SIGNAL_COLUMNS:
        raise ValueError("GPU signal file has an invalid schema")
    parsed_dates = pd.to_datetime(rows["date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("GPU signal file has an invalid date")
    latest_date = parsed_dates.max()
    return rows.loc[parsed_dates.eq(latest_date)].copy()


def _section_status(rows: pd.DataFrame) -> str:
    statuses = {
        _clean(value).upper()
        for value in rows["snapshot_status"]
        if _clean(value)
    }
    for status in (API_KEY_MISSING, SCHEMA_ERROR, PROVIDER_ERROR):
        if status in statuses:
            return status
    if "SUCCESS_WITH_WARNINGS" in statuses:
        return "SUCCESS_WITH_WARNINGS"
    if "PARTIAL" in statuses:
        return "PARTIAL"
    if "SUCCESS" in statuses:
        return "SUCCESS"
    if "NO_MARKET_DATA" in statuses:
        return "NO_MARKET_DATA"
    return PROVIDER_ERROR


def _display_rows(rows: pd.DataFrame) -> list[list[str]]:
    by_model = {
        _clean(row["gpu_model"]): row
        for _, row in rows.iterrows()
        if _clean(row["gpu_model"])
    }
    displayed: list[list[str]] = []
    for model in TRACKED_GPU_MODELS:
        row = by_model.get(model)
        if row is None:
            continue
        has_on_demand_price = _number(
            row["on_demand_median_price_per_gpu_hour"]
        ) is not None
        price_7d = (
            _format_percent(row["rental_price_trend_7d"])
            if has_on_demand_price
            else "N/A"
        )
        price_30d = (
            _format_percent(row["rental_price_trend_30d"])
            if has_on_demand_price
            else "N/A"
        )
        displayed.append(
            [
                model,
                price_7d,
                price_30d,
                _format_count(row["visible_gpu_count"]),
                _format_percent(row["visible_offer_count_trend_7d"]),
                _format_percent(row["visible_offer_count_trend_30d"]),
                _clean(row["supply_signal"]) or "DATA_UNAVAILABLE",
            ]
        )
    return displayed


def _plain_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    output = [
        " | ".join(
            header.ljust(widths[index])
            for index, header in enumerate(headers)
        ),
        "-+-".join("-" * width for width in widths),
    ]
    output.extend(
        " | ".join(
            value.ljust(widths[index])
            for index, value in enumerate(row)
        )
        for row in rows
    )
    return "\n".join(output)


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    header_html = "".join(f"<th>{escape(header)}</th>" for header in headers)
    row_html = "".join(
        "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{row_html}</tbody></table>"


def build_gpu_cloud_email_section(
    signals_file: Path | str = GPU_CLOUD_SIGNALS_FILE,
) -> GPUCloudEmailSection:
    """Read only the formal signal CSV and render its latest daily snapshot."""

    try:
        rows = _read_latest_signals(Path(signals_file))
    except (OSError, ValueError, pd.errors.ParserError):
        return build_gpu_cloud_unavailable_email_section(SCHEMA_ERROR)

    status = _section_status(rows)
    if status in UNAVAILABLE_STATUSES:
        return build_gpu_cloud_unavailable_email_section(status)

    values = _display_rows(rows)
    if not values:
        return build_gpu_cloud_unavailable_email_section(SCHEMA_ERROR)
    headers = [
        "GPU Model",
        "7D Rental Price",
        "30D Rental Price",
        "Visible GPUs",
        "7D Offer Trend",
        "30D Offer Trend",
        "Supply Signal",
    ]
    snapshot = _format_snapshot(
        next(
            (
                value
                for value in rows["source_snapshot_timestamp_utc"]
                if _clean(value)
            ),
            None,
        )
    )
    notes_text = " ".join(_clean(value) for value in rows["data_quality_notes"])
    notes: list[str] = []
    if "gpu_model_warning" in notes_text:
        notes.append(
            "Some A100 memory sizes and H200 form factors could not be "
            "classified precisely."
        )
    if "schema_warning" in notes_text:
        notes.append(
            "Some GPU fields were incomplete; metrics use only verifiable values."
        )

    plain_lines = [
        EMAIL_TITLE,
        "",
        f"Snapshot: {snapshot}",
        f"Scope: {EMAIL_SCOPE}",
        f"Status: {status}",
        "Rental price: on-demand median price per GPU-hour",
        "",
        _plain_table(headers, values),
        "",
        f"Data status: {status}",
        f"Inventory scope: {EMAIL_SCOPE}",
        f"Supply signal scope: {SUPPLY_SIGNAL_SCOPE}",
    ]
    if notes:
        plain_lines.append("Notes: " + " ".join(notes))

    html_parts = [
        f"<section><h2>{escape(EMAIL_TITLE)}</h2>",
        f"<p>Snapshot: {escape(snapshot)}<br>",
        f"Scope: {escape(EMAIL_SCOPE)}<br>",
        f"Status: {escape(status)}<br>",
        "Rental price: on-demand median price per GPU-hour</p>",
        _html_table(headers, values),
        f"<p>Data status: {escape(status)}<br>",
        f"Inventory scope: {escape(EMAIL_SCOPE)}<br>",
        f"Supply signal scope: {escape(SUPPLY_SIGNAL_SCOPE)}</p>",
    ]
    if notes:
        html_parts.append(f"<p>Notes: {escape(' '.join(notes))}</p>")
    html_parts.append("</section>")
    return GPUCloudEmailSection(
        plain_text="\n".join(plain_lines),
        html="".join(html_parts),
        available=True,
        status=status,
    )

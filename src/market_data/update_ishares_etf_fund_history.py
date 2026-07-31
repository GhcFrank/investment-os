"""Maintain official iShares fund history for leadership-industry ETFs."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

from market_data.sector_etf_config import (
    DEFAULT_CONFIG_FILE,
    SectorETF,
    SectorETFConfig,
    load_sector_etf_config,
)
from market_data.update_sector_etf_fund_history import (
    DEFAULT_OUTPUT_DIR,
    FUND_HISTORY_COLUMNS,
    FundHistoryUpdateResult,
    _coerce_fund_history_rows,
    configured_fund_history_path,
    fund_histories_equal,
    load_local_fund_history,
    merge_fund_history,
    validate_state_street_fund_history,
    write_fund_history_atomic,
)
from utils.retry_utils import retry_call, short_error


LOGGER = logging.getLogger(__name__)
SPREADSHEET_NAMESPACE = "urn:schemas-microsoft-com:office:spreadsheet"
SPREADSHEET_NS = {"ss": SPREADSHEET_NAMESPACE}
USER_AGENT = (
    "Mozilla/5.0 (compatible; investment-os/1.0; "
    "+https://www.blackrock.com/)"
)
BARE_AMPERSAND = re.compile(
    r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9A-Fa-f]+;)"
)
ISHARES_HISTORY_HEADERS = {
    "as of": "date",
    "nav per share": "nav",
    "shares outstanding": "shares_outstanding",
}


class ISharesError(RuntimeError):
    """Base class for iShares fund-data failures."""


class ISharesDownloadError(ISharesError):
    """An official iShares/BlackRock document could not be downloaded."""


class ISharesFileFormatError(ISharesError):
    """An official response has an unsupported or unsafe format."""


class ISharesTickerMismatchError(ISharesFileFormatError):
    """The product page identifies a different ticker."""


class ISharesDataValidationError(ISharesError):
    """Parsed iShares fund data violates required constraints."""


@dataclass(frozen=True)
class ISharesDownloadedDocument:
    content: bytes
    content_type: str
    status_code: int


@dataclass(frozen=True)
class ISharesProductSnapshot:
    ticker: str
    snapshot_date: str
    nav: float
    shares_outstanding: int
    total_net_assets: float
    relative_aum_difference: float


@dataclass(frozen=True)
class ISharesETFFundHistorySummary:
    configured_etfs: int
    requested_etfs: int
    succeeded: int
    failed: int
    files_written: int
    files_unchanged: int
    rows_inserted: int
    rows_updated: int
    errors: dict[str, str] = field(default_factory=dict)

    def format(self) -> str:
        lines = [
            "iShares ETF fund-history update summary:",
            f"- configured iShares ETFs: {self.configured_etfs}",
            f"- requested ETFs: {self.requested_etfs}",
            f"- succeeded: {self.succeeded}",
            f"- failed: {self.failed}",
            f"- files written: {self.files_written}",
            f"- files unchanged: {self.files_unchanged}",
            f"- rows inserted: {self.rows_inserted}",
            f"- rows updated: {self.rows_updated}",
        ]
        for ticker, message in self.errors.items():
            lines.append(f"- {ticker} error: {message}")
        return "\n".join(lines)


def build_ishares_fund_download_url(
    config: SectorETFConfig,
    etf: SectorETF,
) -> str:
    """Render the configured official fund-download endpoint."""

    if (
        etf not in config.leadership_overlay_etfs
        or etf.ishares_product_id is None
    ):
        raise ValueError(f"{etf.ticker} is not a configured iShares ETF")
    return config.ishares_fund_download_url_template.format(
        product_id=etf.ishares_product_id,
        ticker=etf.ticker,
        ticker_lower=etf.ticker.lower(),
        product_slug=etf.product_page_slug,
    )


def build_ishares_product_page_url(
    config: SectorETFConfig,
    etf: SectorETF,
) -> str:
    """Render the configured official BlackRock product page."""

    if (
        etf not in config.leadership_overlay_etfs
        or etf.ishares_product_id is None
        or not etf.product_page_slug
    ):
        raise ValueError(f"{etf.ticker} is not a configured iShares ETF")
    return config.ishares_product_page_url_template.format(
        product_id=etf.ishares_product_id,
        product_slug=etf.product_page_slug,
        ticker=etf.ticker,
        ticker_lower=etf.ticker.lower(),
    )


def detect_ishares_download_format(
    content: bytes,
    *,
    content_type: str = "",
) -> str:
    """Identify XLSX or SpreadsheetML by content, never by file extension."""

    if not content:
        raise ISharesFileFormatError("iShares fund download was empty")
    prefix = content[:1024].lstrip(b"\xef\xbb\xbf\x00 \t\r\n").lower()
    lowered_type = content_type.lower()
    if (
        prefix.startswith((b"<html", b"<!doctype html"))
        or b"<html" in prefix
        or b"access denied" in prefix
        or b"captcha" in prefix
        or "text/html" in lowered_type
    ):
        raise ISharesFileFormatError(
            "iShares fund download returned HTML or an access-error page"
        )
    if content.startswith(b"PK"):
        return "xlsx"
    if (
        prefix.startswith(b"<?xml")
        or prefix.startswith(b"<workbook")
        or prefix.startswith(b"<ss:workbook")
    ):
        return "spreadsheetml"
    summary = prefix[:32].decode("ascii", errors="replace")
    raise ISharesFileFormatError(
        "Unknown iShares fund-download format "
        f"(Content-Type={content_type!r}, prefix={summary!r})"
    )


def _download_document(
    url: str,
    *,
    label: str,
    http_client: Any,
    timeout: tuple[int, int],
    max_attempts: int,
    sleep_func: Callable[[float], None],
) -> ISharesDownloadedDocument:
    def request_document() -> ISharesDownloadedDocument:
        try:
            response = http_client.get(
                url,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise ISharesDownloadError(
                f"{label} HTTP request failed: {short_error(error)}"
            ) from error
        except Exception as error:
            raise ISharesDownloadError(
                f"{label} HTTP request failed: {short_error(error)}"
            ) from error
        headers = getattr(response, "headers", {}) or {}
        return ISharesDownloadedDocument(
            content=bytes(getattr(response, "content", b"")),
            content_type=str(headers.get("Content-Type", "")),
            status_code=int(getattr(response, "status_code", 200)),
        )

    return retry_call(
        request_document,
        label=label,
        max_attempts=max_attempts,
        sleep_func=sleep_func,
        logger=LOGGER,
    )


def download_ishares_fund_data(
    url: str,
    *,
    ticker: str,
    http_client: Any = requests,
    timeout: tuple[int, int] = (10, 60),
    max_attempts: int = 3,
    sleep_func: Callable[[float], None] = time.sleep,
) -> ISharesDownloadedDocument:
    document = _download_document(
        url,
        label=f"iShares fund download for {ticker}",
        http_client=http_client,
        timeout=timeout,
        max_attempts=max_attempts,
        sleep_func=sleep_func,
    )
    try:
        detect_ishares_download_format(
            document.content,
            content_type=document.content_type,
        )
    except ISharesFileFormatError:
        LOGGER.error(
            "Invalid iShares fund download for %s: status=%s "
            "Content-Type=%r bytes=%s",
            ticker,
            document.status_code,
            document.content_type,
            len(document.content),
        )
        raise
    return document


def download_ishares_product_page(
    url: str,
    *,
    ticker: str,
    http_client: Any = requests,
    timeout: tuple[int, int] = (10, 60),
    max_attempts: int = 3,
    sleep_func: Callable[[float], None] = time.sleep,
) -> ISharesDownloadedDocument:
    document = _download_document(
        url,
        label=f"BlackRock product page for {ticker}",
        http_client=http_client,
        timeout=timeout,
        max_attempts=max_attempts,
        sleep_func=sleep_func,
    )
    prefix = document.content[:1024].lstrip().lower()
    if (
        not document.content
        or not (
            prefix.startswith((b"<!doctype html", b"<html"))
            or b"<html" in prefix
        )
        or b"access denied" in prefix
        or b"captcha" in prefix
    ):
        raise ISharesFileFormatError(
            f"BlackRock product page for {ticker} is not valid HTML"
        )
    return document


def _normalize_header(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def _rows_to_history(rows: list[list[object]], *, context: str) -> pd.DataFrame:
    header_index: int | None = None
    header_map: dict[int, str] = {}
    for index, row in enumerate(rows):
        normalized = [_normalize_header(value) for value in row]
        candidate = {
            column_index: ISHARES_HISTORY_HEADERS[value]
            for column_index, value in enumerate(normalized)
            if value in ISHARES_HISTORY_HEADERS
        }
        if {"date", "nav", "shares_outstanding"}.issubset(candidate.values()):
            header_index = index
            header_map = candidate
            break
    if header_index is None:
        raise ISharesFileFormatError(
            f"{context} has no Historical date/NAV/shares header"
        )

    records: list[dict[str, object]] = []
    for row in rows[header_index + 1 :]:
        record = {
            canonical: row[column_index] if column_index < len(row) else None
            for column_index, canonical in header_map.items()
        }
        record["total_net_assets"] = pd.NA
        records.append(record)
    if not records:
        raise ISharesFileFormatError(f"{context} Historical sheet is empty")
    raw = pd.DataFrame(records).reindex(columns=FUND_HISTORY_COLUMNS)
    parsed_dates = pd.to_datetime(
        raw["date"],
        errors="coerce",
        format="mixed",
    )
    raw.loc[parsed_dates.notna(), "date"] = parsed_dates.loc[
        parsed_dates.notna()
    ].dt.strftime("%Y-%m-%d")
    for column in ("shares_outstanding", "total_net_assets"):
        missing = (
            raw[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.casefold()
            .isin({"", "-", "--", "n/a", "na"})
        )
        raw.loc[missing, column] = pd.NA
    numeric_shares = pd.to_numeric(
        raw["shares_outstanding"],
        errors="coerce",
    )
    fractional_shares = (
        numeric_shares.notna()
        & numeric_shares.map(lambda value: not float(value).is_integer())
    )
    if fractional_shares.any():
        LOGGER.warning(
            "%s has %s historical non-integer Shares Outstanding row(s); "
            "preserving NAV and normalizing shares to null",
            context,
            int(fractional_shares.sum()),
        )
        raw.loc[fractional_shares, "shares_outstanding"] = pd.NA
    try:
        history = _coerce_fund_history_rows(
            raw,
            context=context,
        )
        validate_state_street_fund_history(history, context=context)
    except Exception as error:
        if isinstance(error, ISharesError):
            raise
        raise ISharesDataValidationError(short_error(error)) from error
    return history


def _spreadsheetml_rows(content: bytes) -> list[list[object]]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ISharesFileFormatError(
            "SpreadsheetML response is not valid UTF-8"
        ) from error
    sanitized = BARE_AMPERSAND.sub("&amp;", text)
    try:
        root = ET.fromstring(sanitized)
    except ET.ParseError as error:
        raise ISharesFileFormatError(
            f"Cannot parse iShares SpreadsheetML: {short_error(error)}"
        ) from error

    name_attribute = f"{{{SPREADSHEET_NAMESPACE}}}Name"
    historical = next(
        (
            worksheet
            for worksheet in root.findall("ss:Worksheet", SPREADSHEET_NS)
            if str(worksheet.get(name_attribute, "")).strip().casefold()
            == "historical"
        ),
        None,
    )
    if historical is None:
        raise ISharesFileFormatError(
            "iShares SpreadsheetML has no Historical worksheet"
        )

    index_attribute = f"{{{SPREADSHEET_NAMESPACE}}}Index"
    rows: list[list[object]] = []
    for xml_row in historical.findall(".//ss:Row", SPREADSHEET_NS):
        values: list[object] = []
        for cell in xml_row.findall("ss:Cell", SPREADSHEET_NS):
            explicit_index = cell.get(index_attribute)
            if explicit_index:
                while len(values) < int(explicit_index) - 1:
                    values.append(None)
            data = cell.find("ss:Data", SPREADSHEET_NS)
            values.append(data.text if data is not None else None)
        rows.append(values)
    return rows


def _xlsx_rows(content: bytes) -> list[list[object]]:
    try:
        with pd.ExcelFile(BytesIO(content), engine="openpyxl") as workbook:
            sheet_name = next(
                (
                    name
                    for name in workbook.sheet_names
                    if str(name).strip().casefold() == "historical"
                ),
                None,
            )
            if sheet_name is None:
                raise ISharesFileFormatError(
                    "iShares XLSX has no Historical worksheet"
                )
            raw = pd.read_excel(
                workbook,
                sheet_name=sheet_name,
                header=None,
            )
    except ISharesFileFormatError:
        raise
    except Exception as error:
        raise ISharesFileFormatError(
            f"Cannot open or read iShares XLSX: {short_error(error)}"
        ) from error
    return raw.where(pd.notna(raw), None).values.tolist()


def parse_ishares_fund_history(
    document: ISharesDownloadedDocument | bytes,
    *,
    ticker: str,
    content_type: str = "",
) -> pd.DataFrame:
    """Parse the official Historical worksheet into the shared CSV schema."""

    if isinstance(document, ISharesDownloadedDocument):
        content = document.content
        content_type = document.content_type
    else:
        content = document
    format_name = detect_ishares_download_format(
        content,
        content_type=content_type,
    )
    rows = (
        _xlsx_rows(content)
        if format_name == "xlsx"
        else _spreadsheetml_rows(content)
    )
    return _rows_to_history(rows, context=f"iShares {ticker}")


def _parse_positive_number(value: str, *, field_name: str) -> float:
    normalized = (
        value.replace("$", "")
        .replace(",", "")
        .replace("\u00a0", " ")
        .strip()
    )
    try:
        number = float(normalized)
    except ValueError as error:
        raise ISharesDataValidationError(
            f"iShares {field_name} is not numeric: {value!r}"
        ) from error
    if not math.isfinite(number) or number <= 0:
        raise ISharesDataValidationError(
            f"iShares {field_name} must be finite and positive"
        )
    return number


def _parse_as_of_date(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if normalized.casefold().startswith("as of "):
        normalized = normalized[6:].strip()
    for date_format in ("%b %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(normalized, date_format).date().isoformat()
        except ValueError:
            continue
    raise ISharesDataValidationError(
        f"iShares {field_name} has invalid as-of date: {value!r}"
    )


def _iter_json_nodes(value: object):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _iter_json_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_json_nodes(child)


def parse_ishares_product_snapshot(
    html_content: bytes | str,
    *,
    requested_ticker: str,
) -> ISharesProductSnapshot:
    """Extract official NAV, AUM, shares, and their individual dates."""

    if isinstance(html_content, bytes):
        try:
            html = html_content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ISharesFileFormatError(
                "BlackRock product page is not UTF-8 HTML"
            ) from error
    else:
        html = html_content
    soup = BeautifulSoup(html, "html.parser")

    ticker: str | None = None
    nav_value: str | None = None
    nav_date: str | None = None
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            payload = json.loads(script.string or script.get_text())
        except (json.JSONDecodeError, TypeError):
            continue
        for node in _iter_json_nodes(payload):
            if not isinstance(node, dict):
                continue
            if (
                str(node.get("propertyID", "")).casefold() == "ticker"
                and node.get("value")
            ):
                ticker = str(node["value"]).strip().upper()
            if (
                str(node.get("name", "")).casefold() == "nav as of"
                and node.get("value") is not None
            ):
                nav_value = str(node["value"])
                reference = node.get("valueReference")
                if isinstance(reference, dict) and reference.get("value"):
                    nav_date = str(reference["value"])

    expected_ticker = requested_ticker.strip().upper()
    if ticker != expected_ticker:
        raise ISharesTickerMismatchError(
            f"Requested {expected_ticker}, product page declares "
            f"{ticker or 'no ticker'}"
        )
    if nav_value is None or nav_date is None:
        raise ISharesFileFormatError(
            f"BlackRock product page for {expected_ticker} has no dated NAV"
        )

    def data_text(data_id: str) -> str:
        element = soup.select_one(f'[data-id="{data_id}"]')
        if element is None:
            raise ISharesFileFormatError(
                f"BlackRock product page is missing {data_id}"
            )
        return element.get_text(" ", strip=True)

    assets_value = data_text("keyFundFacts-totalNetAssetsFundLevel-data")
    assets_date = data_text("keyFundFacts-totalNetAssetsFundLevel-asOf")
    shares_value = data_text("keyFundFacts-sharesOutstanding-data")
    shares_date = data_text("keyFundFacts-sharesOutstanding-asOf")

    parsed_nav_date = _parse_as_of_date(nav_date, field_name="NAV")
    parsed_assets_date = _parse_as_of_date(
        assets_date,
        field_name="Net Assets of Fund",
    )
    parsed_shares_date = _parse_as_of_date(
        shares_date,
        field_name="Shares Outstanding",
    )
    dates = {parsed_nav_date, parsed_assets_date, parsed_shares_date}
    if len(dates) != 1:
        raise ISharesDataValidationError(
            f"iShares {expected_ticker} field dates do not match: "
            f"NAV={parsed_nav_date}, AUM={parsed_assets_date}, "
            f"shares={parsed_shares_date}"
        )

    nav = _parse_positive_number(nav_value, field_name="NAV")
    assets = _parse_positive_number(
        assets_value,
        field_name="Net Assets of Fund",
    )
    shares_float = _parse_positive_number(
        shares_value,
        field_name="Shares Outstanding",
    )
    if not shares_float.is_integer():
        raise ISharesDataValidationError(
            "iShares Shares Outstanding must be integer-compatible"
        )
    shares = int(shares_float)
    expected_assets = nav * shares
    relative_difference = abs(assets - expected_assets) / max(
        assets,
        expected_assets,
    )
    if relative_difference > 0.02:
        raise ISharesDataValidationError(
            f"iShares {expected_ticker} official AUM differs from NAV x "
            f"shares by {relative_difference:.2%}"
        )
    if relative_difference > 0.005:
        LOGGER.warning(
            "iShares %s official AUM differs from rounded product-page "
            "NAV x shares by %.2f%%",
            expected_ticker,
            relative_difference * 100,
        )

    return ISharesProductSnapshot(
        ticker=expected_ticker,
        snapshot_date=dates.pop(),
        nav=nav,
        shares_outstanding=shares,
        total_net_assets=assets,
        relative_aum_difference=relative_difference,
    )


def enrich_history_with_official_snapshot(
    history: pd.DataFrame,
    snapshot: ISharesProductSnapshot,
) -> pd.DataFrame:
    """Attach official AUM only to the matching official historical row."""

    matches = history["date"].eq(snapshot.snapshot_date)
    if int(matches.sum()) != 1:
        raise ISharesDataValidationError(
            f"iShares {snapshot.ticker} Historical worksheet has no unique "
            f"row for product-page date {snapshot.snapshot_date}"
        )
    historical_nav = float(history.loc[matches, "nav"].iloc[0])
    historical_shares = int(history.loc[matches, "shares_outstanding"].iloc[0])
    if not math.isclose(
        historical_nav,
        snapshot.nav,
        rel_tol=0,
        abs_tol=0.011,
    ):
        raise ISharesDataValidationError(
            f"iShares {snapshot.ticker} product-page NAV "
            f"{snapshot.nav} does not match exact Historical NAV "
            f"{historical_nav}"
        )
    if historical_shares != snapshot.shares_outstanding:
        raise ISharesDataValidationError(
            f"iShares {snapshot.ticker} product-page shares "
            f"{snapshot.shares_outstanding} do not match Historical shares "
            f"{historical_shares}"
        )

    output = history.copy()
    output.loc[matches, "total_net_assets"] = snapshot.total_net_assets
    quality = validate_state_street_fund_history(
        output,
        context=f"iShares {snapshot.ticker}",
    )
    if quality.consistency_warning_rows:
        LOGGER.warning(
            "iShares %s has %s NAV x shares consistency warning row(s)",
            snapshot.ticker,
            quality.consistency_warning_rows,
        )
    return output


def update_one_ishares_etf(
    etf: SectorETF,
    config: SectorETFConfig,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    http_client: Any = requests,
    max_attempts: int = 3,
    sleep_func: Callable[[float], None] = time.sleep,
) -> FundHistoryUpdateResult:
    """Download, validate, merge, and atomically write one iShares ETF."""

    output_path = configured_fund_history_path(output_dir, etf)
    local = load_local_fund_history(output_path)
    fund_document = download_ishares_fund_data(
        build_ishares_fund_download_url(config, etf),
        ticker=etf.ticker,
        http_client=http_client,
        max_attempts=max_attempts,
        sleep_func=sleep_func,
    )
    product_document = download_ishares_product_page(
        build_ishares_product_page_url(config, etf),
        ticker=etf.ticker,
        http_client=http_client,
        max_attempts=max_attempts,
        sleep_func=sleep_func,
    )
    remote = parse_ishares_fund_history(
        fund_document,
        ticker=etf.ticker,
    )
    snapshot = parse_ishares_product_snapshot(
        product_document.content,
        requested_ticker=etf.ticker,
    )
    remote = enrich_history_with_official_snapshot(remote, snapshot)

    if not local.empty:
        local_latest = date.fromisoformat(str(local["date"].max()))
        remote_latest = date.fromisoformat(str(remote["date"].max()))
        if remote_latest < local_latest:
            lag_days = (local_latest - remote_latest).days
            if lag_days > 7:
                raise ISharesDataValidationError(
                    f"iShares {etf.ticker} latest date {remote_latest} is "
                    f"{lag_days} days older than local {local_latest}"
                )
            LOGGER.warning(
                "iShares %s latest date %s is behind local %s; newer local "
                "rows will be preserved",
                etf.ticker,
                remote_latest,
                local_latest,
            )

    merge_result = merge_fund_history(local, remote)
    changed = (
        not output_path.exists()
        or bool(local.attrs.get("needs_rewrite", False))
        or not fund_histories_equal(local, merge_result.history)
    )
    if changed:
        write_fund_history_atomic(merge_result.history, output_path)
    return FundHistoryUpdateResult(
        ticker=etf.ticker,
        rows=len(merge_result.history),
        inserted_rows=merge_result.inserted_rows,
        updated_rows=merge_result.updated_rows,
        file_written=changed,
        earliest_date=str(merge_result.history["date"].min()),
        latest_date=str(merge_result.history["date"].max()),
        consistency_warning_rows=0,
        severe_consistency_rows=0,
        zero_share_rows_normalized=0,
    )


def _select_ishares_etfs(
    config: SectorETFConfig,
    tickers: Sequence[str] | None,
) -> tuple[SectorETF, ...]:
    if not tickers:
        return config.leadership_overlay_etfs
    requested = {
        str(ticker).strip().upper()
        for ticker in tickers
        if str(ticker).strip()
    }
    configured = {etf.ticker for etf in config.leadership_overlay_etfs}
    invalid = requested - configured
    if invalid:
        raise ValueError(
            "Unconfigured iShares ETF ticker(s): "
            + ", ".join(sorted(invalid))
        )
    return tuple(
        etf
        for etf in config.leadership_overlay_etfs
        if etf.ticker in requested
    )


def run_ishares_etf_fund_history_update(
    *,
    config_path: Path | str = DEFAULT_CONFIG_FILE,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    tickers: Sequence[str] | None = None,
    http_client: Any = requests,
    max_attempts: int = 3,
    sleep_func: Callable[[float], None] = time.sleep,
) -> ISharesETFFundHistorySummary:
    """Update both official iShares histories with per-ticker isolation."""

    config = load_sector_etf_config(config_path)
    selected_etfs = _select_ishares_etfs(config, tickers)
    if not selected_etfs:
        raise ValueError("At least one configured iShares ticker is required")
    history_dir = Path(output_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    if not history_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {history_dir}")

    results: list[FundHistoryUpdateResult] = []
    errors: dict[str, str] = {}
    for etf in selected_etfs:
        try:
            result = update_one_ishares_etf(
                etf,
                config,
                output_dir=history_dir,
                http_client=http_client,
                max_attempts=max_attempts,
                sleep_func=sleep_func,
            )
            results.append(result)
            LOGGER.info(
                "Updated iShares %s history: rows=%s inserted=%s "
                "updated=%s written=%s",
                etf.ticker,
                result.rows,
                result.inserted_rows,
                result.updated_rows,
                result.file_written,
            )
        except Exception as error:
            message = short_error(error)
            errors[etf.ticker] = message
            LOGGER.error(
                "iShares fund history failed for %s: %s",
                etf.ticker,
                message,
            )

    return ISharesETFFundHistorySummary(
        configured_etfs=len(config.leadership_overlay_etfs),
        requested_etfs=len(selected_etfs),
        succeeded=len(results),
        failed=len(errors),
        files_written=sum(result.file_written for result in results),
        files_unchanged=sum(not result.file_written for result in results),
        rows_inserted=sum(result.inserted_rows for result in results),
        rows_updated=sum(result.updated_rows for result in results),
        errors=errors,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Update official iShares NAV, shares, and latest official AUM "
            "history for leadership-industry ETFs."
        )
    )
    parser.add_argument(
        "--tickers",
        help="Comma-separated configured iShares tickers.",
    )
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    tickers = args.tickers.split(",") if args.tickers else None
    try:
        summary = run_ishares_etf_fund_history_update(tickers=tickers)
    except Exception as error:
        print(
            f"iShares fund-history update failed: {short_error(error)}",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    print(summary.format())
    if summary.succeeded == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""
Maintain official State Street NAV and fund-size history for sector ETFs.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.market_data.sector_etf_config import (
    BASE_DIR,
    DEFAULT_CONFIG_FILE,
    SectorETF,
    SectorETFConfig,
    load_sector_etf_config,
    validate_fund_history_filename,
)
from src.utils.csv_utils import atomic_write_csv
from src.utils.date_utils import MARKET_TIMEZONE
from src.utils.retry_utils import retry_call, short_error


LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = (
    BASE_DIR / "data" / "market_data" / "sector_etf_fund_history"
)
FUND_HISTORY_COLUMNS = [
    "date",
    "nav",
    "shares_outstanding",
    "total_net_assets",
]
STATE_STREET_SHEET_NAME = "navhist"
REMOTE_COLUMNS = {
    "date": "date",
    "nav": "nav",
    "shares outstanding": "shares_outstanding",
    "total net assets": "total_net_assets",
}


class StateStreetError(RuntimeError):
    """Base class for State Street fund-history failures."""


class StateStreetDownloadError(StateStreetError):
    """The remote workbook could not be downloaded."""


class StateStreetFileFormatError(StateStreetError):
    """The response is not the expected State Street workbook."""


class StateStreetTickerMismatchError(StateStreetFileFormatError):
    """The workbook ticker does not match the requested ETF."""


class StateStreetDataValidationError(StateStreetError):
    """Parsed or local fund-history data violates required constraints."""


@dataclass(frozen=True)
class FundHistoryQuality:
    consistency_warning_rows: int = 0
    severe_consistency_rows: int = 0
    zero_share_rows_normalized: int = 0


@dataclass(frozen=True)
class FundHistoryMergeResult:
    history: pd.DataFrame
    inserted_rows: int
    updated_rows: int


@dataclass(frozen=True)
class FundHistoryMigrationResult:
    ticker: str
    action: str
    legacy_path: Path
    output_path: Path
    inserted_rows: int = 0
    updated_rows: int = 0


@dataclass(frozen=True)
class FundHistoryUpdateResult:
    ticker: str
    rows: int
    inserted_rows: int
    updated_rows: int
    file_written: bool
    earliest_date: str
    latest_date: str
    consistency_warning_rows: int
    severe_consistency_rows: int
    zero_share_rows_normalized: int


@dataclass(frozen=True)
class SectorETFFundHistorySummary:
    configured_etfs: int
    requested_etfs: int
    succeeded: int
    failed: int
    files_written: int
    files_unchanged: int
    rows_inserted: int
    rows_updated: int
    consistency_warning_rows: int
    severe_consistency_rows: int
    zero_share_rows_normalized: int
    mode: str
    errors: dict[str, str] = field(default_factory=dict)

    def format(self) -> str:
        lines = [
            "State Street sector ETF fund-history update summary:",
            f"- mode: {self.mode}",
            f"- configured ETFs: {self.configured_etfs}",
            f"- requested ETFs: {self.requested_etfs}",
            f"- succeeded: {self.succeeded}",
            f"- failed: {self.failed}",
            f"- files written: {self.files_written}",
            f"- files unchanged: {self.files_unchanged}",
            f"- rows inserted: {self.rows_inserted}",
            f"- rows updated: {self.rows_updated}",
            (
                "- AUM consistency warning rows: "
                f"{self.consistency_warning_rows}"
            ),
            (
                "- severe consistency rows: "
                f"{self.severe_consistency_rows}"
            ),
            (
                "- zero-share rows normalized to null: "
                f"{self.zero_share_rows_normalized}"
            ),
        ]
        for ticker, message in self.errors.items():
            lines.append(f"- {ticker} error: {message}")
        return "\n".join(lines)


def build_state_street_nav_history_url(
    config: SectorETFConfig,
    ticker: str,
) -> str:
    """
    Build a configured State Street URL without hard-coded per-ETF paths.
    """

    normalized_ticker = ticker.strip().upper()
    configured = {etf.ticker for etf in config.etfs}
    if normalized_ticker not in configured:
        raise ValueError(f"Ticker is not configured: {normalized_ticker}")
    return config.state_street_nav_history_url_template.format(
        ticker_lower=normalized_ticker.lower()
    )


def _validate_xlsx_payload(
    content: bytes,
    *,
    content_type: str = "",
) -> None:
    if not content:
        raise StateStreetFileFormatError("State Street response was empty")
    lowered_type = content_type.lower()
    if "html" in lowered_type or "text/plain" in lowered_type:
        raise StateStreetFileFormatError(
            f"State Street returned unexpected Content-Type: {content_type}"
        )
    if content.lstrip().startswith((b"<html", b"<!doctype", b"<HTML")):
        raise StateStreetFileFormatError(
            "State Street returned an HTML document instead of XLSX"
        )
    if not content.startswith(b"PK"):
        raise StateStreetFileFormatError(
            "State Street response does not have an XLSX ZIP signature"
        )


def download_state_street_nav_history(
    url: str,
    *,
    ticker: str,
    http_client: Any = requests,
    timeout: tuple[int, int] = (10, 60),
    max_attempts: int = 3,
    sleep_func: Callable[[float], None] = time.sleep,
) -> bytes:
    """
    Download and validate an XLSX payload without writing it to disk.
    """

    def request_workbook() -> bytes:
        try:
            response = http_client.get(url, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as error:
            raise StateStreetDownloadError(
                f"State Street HTTP request failed: {short_error(error)}"
            ) from error
        except Exception as error:
            raise StateStreetDownloadError(
                f"State Street HTTP request failed: {short_error(error)}"
            ) from error

        content = bytes(getattr(response, "content", b""))
        headers = getattr(response, "headers", {}) or {}
        _validate_xlsx_payload(
            content,
            content_type=str(headers.get("Content-Type", "")),
        )
        return content

    return retry_call(
        request_workbook,
        label=f"State Street fund history for {ticker}",
        max_attempts=max_attempts,
        sleep_func=sleep_func,
        logger=LOGGER,
    )


def _normalize_header(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def _parse_date_value(value: object) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return pd.Timestamp(value).date().isoformat()

    text = str(value).strip()
    if not text:
        return None
    for date_format in ("%d-%b-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, date_format).date().isoformat()
        except ValueError:
            continue
    return None


def _parse_optional_number(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text == "-":
        return None
    text = text.replace(",", "").replace("$", "")
    try:
        numeric = float(text)
    except ValueError:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _is_missing_marker(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() in {"", "-"}


def _coerce_fund_history_rows(
    raw: pd.DataFrame,
    *,
    context: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    zero_share_rows = 0
    for row_number, row in raw.iterrows():
        parsed_date = _parse_date_value(row.get("date"))
        if parsed_date is None:
            continue

        nav = _parse_optional_number(row.get("nav"))
        if nav is None:
            raise StateStreetDataValidationError(
                f"{context} has a dated row without numeric NAV at row "
                f"{row_number}"
            )

        shares_raw = row.get("shares_outstanding")
        shares = _parse_optional_number(shares_raw)
        if shares is None and not _is_missing_marker(shares_raw):
            raise StateStreetDataValidationError(
                f"{context} has invalid Shares Outstanding at row {row_number}"
            )
        if shares is not None and not float(shares).is_integer():
            raise StateStreetDataValidationError(
                f"{context} has non-integer Shares Outstanding at row "
                f"{row_number}"
            )
        if shares is not None and shares < 0:
            raise StateStreetDataValidationError(
                f"{context} has negative Shares Outstanding at row {row_number}"
            )
        if shares == 0:
            zero_share_rows += 1
            LOGGER.warning(
                "%s has zero Shares Outstanding on %s; preserving NAV and "
                "normalizing shares to null",
                context,
                parsed_date,
            )
            shares = None

        assets_raw = row.get("total_net_assets")
        assets = _parse_optional_number(assets_raw)
        if assets is None and not _is_missing_marker(assets_raw):
            raise StateStreetDataValidationError(
                f"{context} has invalid Total Net Assets at row {row_number}"
            )

        rows.append(
            {
                "date": parsed_date,
                "nav": nav,
                "shares_outstanding": (
                    int(shares) if shares is not None else pd.NA
                ),
                "total_net_assets": (
                    assets if assets is not None else pd.NA
                ),
            }
        )

    if not rows:
        raise StateStreetDataValidationError(
            f"{context} contains no valid date/NAV rows"
        )

    output = pd.DataFrame(rows, columns=FUND_HISTORY_COLUMNS)
    output["nav"] = pd.to_numeric(output["nav"], errors="raise").astype(float)
    output["shares_outstanding"] = pd.array(
        output["shares_outstanding"],
        dtype="Int64",
    )
    output["total_net_assets"] = pd.to_numeric(
        output["total_net_assets"],
        errors="coerce",
    ).astype(float)
    output = output.sort_values("date", kind="stable").reset_index(drop=True)
    output.attrs["zero_share_rows_normalized"] = zero_share_rows
    return output


def parse_state_street_nav_history(
    workbook_content: bytes,
    *,
    requested_ticker: str,
) -> pd.DataFrame:
    """
    Parse the real State Street navhist layout from an in-memory XLSX.
    """

    _validate_xlsx_payload(workbook_content)
    try:
        with pd.ExcelFile(
            BytesIO(workbook_content),
            engine="openpyxl",
        ) as workbook:
            if STATE_STREET_SHEET_NAME not in workbook.sheet_names:
                raise StateStreetFileFormatError(
                    "State Street XLSX does not contain the navhist sheet"
                )
            metadata = pd.read_excel(
                workbook,
                sheet_name=STATE_STREET_SHEET_NAME,
                header=None,
                usecols="A:B",
                nrows=2,
            )
            raw_history = pd.read_excel(
                workbook,
                sheet_name=STATE_STREET_SHEET_NAME,
                header=3,
                usecols="A:D",
            )
    except StateStreetFileFormatError:
        raise
    except Exception as error:
        raise StateStreetFileFormatError(
            f"Cannot open or read State Street XLSX: {short_error(error)}"
        ) from error

    if metadata.shape[0] < 2 or metadata.shape[1] < 2:
        raise StateStreetFileFormatError(
            "State Street navhist metadata rows are missing"
        )
    ticker_label = _normalize_header(metadata.iat[1, 0]).rstrip(":")
    workbook_ticker = str(metadata.iat[1, 1]).strip().upper()
    if ticker_label != "ticker symbol" or not workbook_ticker:
        raise StateStreetFileFormatError(
            "State Street navhist ticker metadata is invalid"
        )
    expected_ticker = requested_ticker.strip().upper()
    if workbook_ticker != expected_ticker:
        raise StateStreetTickerMismatchError(
            f"Requested {expected_ticker}, workbook declares {workbook_ticker}"
        )

    renamed: dict[object, str] = {}
    for column in raw_history.columns:
        normalized = _normalize_header(column)
        if normalized in REMOTE_COLUMNS:
            renamed[column] = REMOTE_COLUMNS[normalized]
    raw_history = raw_history.rename(columns=renamed)
    missing_columns = [
        column
        for column in FUND_HISTORY_COLUMNS
        if column not in raw_history.columns
    ]
    if missing_columns:
        raise StateStreetFileFormatError(
            "State Street navhist is missing required columns: "
            + ", ".join(missing_columns)
        )

    parsed = _coerce_fund_history_rows(
        raw_history.reindex(columns=FUND_HISTORY_COLUMNS),
        context=f"State Street {expected_ticker}",
    )
    validate_state_street_fund_history(
        parsed,
        context=f"State Street {expected_ticker}",
    )
    return parsed


def validate_state_street_fund_history(
    history: pd.DataFrame,
    *,
    context: str = "fund history",
    today: date | None = None,
) -> FundHistoryQuality:
    """
    Enforce primary-key, numeric, date, and NAV/AUM consistency constraints.
    """

    if list(history.columns) != FUND_HISTORY_COLUMNS:
        raise StateStreetDataValidationError(
            f"{context} schema must be {FUND_HISTORY_COLUMNS}"
        )
    if history.empty:
        raise StateStreetDataValidationError(f"{context} is empty")
    if history["date"].isna().any() or history["date"].duplicated().any():
        raise StateStreetDataValidationError(
            f"{context} contains missing or duplicate dates"
        )

    dates = pd.to_datetime(history["date"], errors="coerce")
    if dates.isna().any():
        raise StateStreetDataValidationError(
            f"{context} contains invalid dates"
        )
    current_date = today or datetime.now(MARKET_TIMEZONE).date()
    if dates.dt.date.gt(current_date).any():
        raise StateStreetDataValidationError(
            f"{context} contains future dates"
        )

    nav = pd.to_numeric(history["nav"], errors="coerce")
    if nav.isna().any() or nav.le(0).any():
        raise StateStreetDataValidationError(
            f"{context} NAV must be greater than zero"
        )
    shares = pd.to_numeric(
        history["shares_outstanding"],
        errors="coerce",
    )
    if shares.dropna().le(0).any():
        raise StateStreetDataValidationError(
            f"{context} Shares Outstanding must be positive or null"
        )
    if (
        shares.dropna()
        .map(lambda value: float(value).is_integer())
        .eq(False)
        .any()
    ):
        raise StateStreetDataValidationError(
            f"{context} Shares Outstanding must be integer-compatible"
        )
    assets = pd.to_numeric(history["total_net_assets"], errors="coerce")
    if assets.dropna().lt(0).any():
        raise StateStreetDataValidationError(
            f"{context} Total Net Assets must be non-negative or null"
        )

    complete = nav.notna() & shares.notna() & assets.notna()
    zero_share_rows = int(
        history.attrs.get("zero_share_rows_normalized", 0)
    )
    if not complete.any():
        return FundHistoryQuality(
            zero_share_rows_normalized=zero_share_rows
        )
    expected_assets = nav[complete] * shares[complete]
    denominator = pd.concat(
        [expected_assets.abs(), assets[complete].abs()],
        axis=1,
    ).max(axis=1)
    relative_difference = (
        (assets[complete] - expected_assets).abs()
        / denominator.where(denominator.ne(0), 1)
    )
    warning_rows = int(relative_difference.gt(0.001).sum())
    severe_rows = int(relative_difference.gt(0.01).sum())
    return FundHistoryQuality(
        consistency_warning_rows=warning_rows,
        severe_consistency_rows=severe_rows,
        zero_share_rows_normalized=zero_share_rows,
    )


def _empty_fund_history() -> pd.DataFrame:
    return pd.DataFrame(columns=FUND_HISTORY_COLUMNS)


def load_local_fund_history(path: Path | str) -> pd.DataFrame:
    """
    Load a local per-ETF CSV and remember whether canonical rewrite is needed.
    """

    csv_path = Path(path)
    if not csv_path.exists():
        output = _empty_fund_history()
        output.attrs["needs_rewrite"] = False
        return output
    try:
        raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        output = _empty_fund_history()
        output.attrs["needs_rewrite"] = True
        return output
    if list(raw.columns) != FUND_HISTORY_COLUMNS:
        raise StateStreetDataValidationError(
            f"Local CSV schema mismatch for {csv_path}: "
            f"found {list(raw.columns)}"
        )
    if raw.empty:
        output = _empty_fund_history()
        output.attrs["needs_rewrite"] = True
        return output

    normalized = _coerce_fund_history_rows(
        raw,
        context=f"local file {csv_path}",
    )
    validate_state_street_fund_history(
        normalized,
        context=f"local file {csv_path}",
    )
    original_dates = [
        str(value).strip()
        for value in raw["date"]
    ]
    normalized_dates = normalized["date"].tolist()
    normalized.attrs["needs_rewrite"] = (
        original_dates != normalized_dates
        or len(set(original_dates)) != len(original_dates)
    )
    return normalized


def _values_differ(left: object, right: object) -> bool:
    left_missing = pd.isna(left)
    right_missing = pd.isna(right)
    if left_missing and right_missing:
        return False
    if left_missing != right_missing:
        return True
    return not math.isclose(
        float(left),
        float(right),
        rel_tol=1e-12,
        abs_tol=1e-9,
    )


def fund_histories_equal(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> bool:
    if len(left) != len(right):
        return False
    if left["date"].tolist() != right["date"].tolist():
        return False
    for column in FUND_HISTORY_COLUMNS[1:]:
        if any(
            _values_differ(left_value, right_value)
            for left_value, right_value in zip(left[column], right[column])
        ):
            return False
    return True


def merge_fund_history(
    local: pd.DataFrame,
    remote: pd.DataFrame,
) -> FundHistoryMergeResult:
    """
    Merge by date, preferring remote non-null fields over local values.
    """

    if local.empty:
        merged = remote.copy()
        merged.attrs.clear()
        return FundHistoryMergeResult(
            history=merged,
            inserted_rows=len(remote),
            updated_rows=0,
        )

    validate_state_street_fund_history(local, context="local fund history")
    validate_state_street_fund_history(remote, context="remote fund history")
    local_indexed = local.set_index("date")
    remote_indexed = remote.set_index("date")
    all_dates = local_indexed.index.union(remote_indexed.index).sort_values()
    merged = local_indexed.reindex(all_dates)
    updated_rows = 0

    shared_dates = local_indexed.index.intersection(remote_indexed.index)
    for remote_date in shared_dates:
        if any(
            not pd.isna(remote_indexed.at[remote_date, column])
            and _values_differ(
                local_indexed.at[remote_date, column],
                remote_indexed.at[remote_date, column],
            )
            for column in FUND_HISTORY_COLUMNS[1:]
        ):
            updated_rows += 1

    for column in FUND_HISTORY_COLUMNS[1:]:
        remote_values = remote_indexed[column].reindex(all_dates)
        merged[column] = remote_values.combine_first(merged[column])

    merged = merged.reset_index(names="date")
    merged["nav"] = pd.to_numeric(merged["nav"], errors="raise").astype(float)
    merged["shares_outstanding"] = pd.array(
        merged["shares_outstanding"],
        dtype="Int64",
    )
    merged["total_net_assets"] = pd.to_numeric(
        merged["total_net_assets"],
        errors="coerce",
    ).astype(float)
    merged = merged.reindex(columns=FUND_HISTORY_COLUMNS)
    validate_state_street_fund_history(merged, context="merged fund history")
    return FundHistoryMergeResult(
        history=merged,
        inserted_rows=len(remote_indexed.index.difference(local_indexed.index)),
        updated_rows=updated_rows,
    )


def write_fund_history_atomic(
    history: pd.DataFrame,
    path: Path | str,
) -> None:
    validate_state_street_fund_history(history)
    atomic_write_csv(history, Path(path), FUND_HISTORY_COLUMNS)


def configured_fund_history_path(
    output_dir: Path | str,
    etf: SectorETF,
) -> Path:
    """
    Resolve the validated, configured CSV basename within the output directory.
    """

    filename = validate_fund_history_filename(
        etf.fund_history_filename,
        location=f"{etf.ticker}.fund_history_filename",
    )
    return Path(output_dir) / filename


def legacy_fund_history_path(
    output_dir: Path | str,
    etf: SectorETF,
) -> Path:
    """
    Resolve the former ticker-only path for migration reads/removal only.
    """

    return Path(output_dir) / (etf.ticker.casefold() + ".csv")


def migrate_one_sector_etf_fund_history(
    etf: SectorETF,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> FundHistoryMigrationResult:
    """
    Safely migrate one legacy ticker-only CSV to its configured filename.

    If both paths exist, configured-file non-null values take precedence while
    legacy values fill configured nulls. The legacy file is removed only after
    an atomic configured-file write succeeds and the written result validates.
    """

    output_path = configured_fund_history_path(output_dir, etf)
    legacy_path = legacy_fund_history_path(output_dir, etf)
    if output_path == legacy_path:
        raise ValueError(
            f"Configured fund-history filename for {etf.ticker} must differ "
            "from its legacy ticker-only filename"
        )
    output_exists = output_path.exists()
    legacy_exists = legacy_path.exists()

    if not legacy_exists and not output_exists:
        return FundHistoryMigrationResult(
            ticker=etf.ticker,
            action="absent",
            legacy_path=legacy_path,
            output_path=output_path,
        )

    if not legacy_exists:
        load_local_fund_history(output_path)
        return FundHistoryMigrationResult(
            ticker=etf.ticker,
            action="validated",
            legacy_path=legacy_path,
            output_path=output_path,
        )

    legacy = load_local_fund_history(legacy_path)
    if not output_exists:
        legacy_path.replace(output_path)
        return FundHistoryMigrationResult(
            ticker=etf.ticker,
            action="renamed",
            legacy_path=legacy_path,
            output_path=output_path,
            inserted_rows=len(legacy),
        )

    configured = load_local_fund_history(output_path)
    merge_result = merge_fund_history(legacy, configured)
    write_fund_history_atomic(merge_result.history, output_path)
    written = load_local_fund_history(output_path)
    if not fund_histories_equal(written, merge_result.history):
        raise StateStreetDataValidationError(
            f"Migrated CSV verification failed for {etf.ticker}"
        )
    legacy_path.unlink()
    return FundHistoryMigrationResult(
        ticker=etf.ticker,
        action="merged",
        legacy_path=legacy_path,
        output_path=output_path,
        inserted_rows=merge_result.inserted_rows,
        updated_rows=merge_result.updated_rows,
    )


def migrate_sector_etf_fund_history_files(
    config: SectorETFConfig,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    etfs: Sequence[SectorETF] | None = None,
) -> tuple[FundHistoryMigrationResult, ...]:
    """
    Migrate and validate configured ETF files without any network access.
    """

    selected_etfs = tuple(etfs) if etfs is not None else config.etfs
    return tuple(
        migrate_one_sector_etf_fund_history(
            etf,
            output_dir=output_dir,
        )
        for etf in selected_etfs
    )


def update_one_sector_etf_fund_history(
    etf: SectorETF,
    config: SectorETFConfig,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    http_client: Any = requests,
    max_attempts: int = 3,
    sleep_func: Callable[[float], None] = time.sleep,
) -> FundHistoryUpdateResult:
    """
    Download, parse, merge, and conditionally write one ETF history.
    """

    output_path = configured_fund_history_path(output_dir, etf)
    local = load_local_fund_history(output_path)
    url = build_state_street_nav_history_url(config, etf.ticker)
    workbook_content = download_state_street_nav_history(
        url,
        ticker=etf.ticker,
        http_client=http_client,
        max_attempts=max_attempts,
        sleep_func=sleep_func,
    )
    remote = parse_state_street_nav_history(
        workbook_content,
        requested_ticker=etf.ticker,
    )
    quality = validate_state_street_fund_history(
        remote,
        context=f"State Street {etf.ticker}",
    )
    if quality.consistency_warning_rows:
        LOGGER.warning(
            "State Street %s has %s NAV x shares consistency warning "
            "row(s), %s severe",
            etf.ticker,
            quality.consistency_warning_rows,
            quality.severe_consistency_rows,
        )

    if not local.empty:
        local_latest = date.fromisoformat(local["date"].max())
        remote_latest = date.fromisoformat(remote["date"].max())
        if remote_latest < local_latest:
            lag_days = (local_latest - remote_latest).days
            if lag_days > 7:
                raise StateStreetDataValidationError(
                    f"State Street {etf.ticker} latest date {remote_latest} is "
                    f"{lag_days} days older than local {local_latest}"
                )
            LOGGER.warning(
                "State Street %s latest date %s is behind local %s; local "
                "newer rows will be preserved",
                etf.ticker,
                remote_latest,
                local_latest,
            )

    merge_result = merge_fund_history(local, remote)
    needs_rewrite = bool(local.attrs.get("needs_rewrite", False))
    changed = (
        not output_path.exists()
        or needs_rewrite
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
        earliest_date=merge_result.history["date"].min(),
        latest_date=merge_result.history["date"].max(),
        consistency_warning_rows=quality.consistency_warning_rows,
        severe_consistency_rows=quality.severe_consistency_rows,
        zero_share_rows_normalized=quality.zero_share_rows_normalized,
    )


def _select_etfs(
    config: SectorETFConfig,
    tickers: Sequence[str] | None,
) -> tuple[SectorETF, ...]:
    if not tickers:
        return config.etfs
    requested = {
        ticker.strip().upper()
        for ticker in tickers
        if ticker.strip()
    }
    configured = {etf.ticker for etf in config.etfs}
    invalid = requested - configured
    if invalid:
        raise ValueError(
            "Unconfigured sector ETF ticker(s): "
            + ", ".join(sorted(invalid))
        )
    return tuple(etf for etf in config.etfs if etf.ticker in requested)


def run_sector_etf_fund_history_update(
    *,
    config_path: Path | str = DEFAULT_CONFIG_FILE,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    tickers: Sequence[str] | None = None,
    bootstrap: bool = False,
    full_refresh: bool = False,
    http_client: Any = requests,
    max_attempts: int = 3,
    sleep_func: Callable[[float], None] = time.sleep,
) -> SectorETFFundHistorySummary:
    """
    Update selected State Street histories with per-ETF failure isolation.
    """

    if bootstrap and full_refresh:
        raise ValueError("bootstrap and full_refresh cannot be used together")
    config = load_sector_etf_config(config_path)
    selected_etfs = _select_etfs(config, tickers)
    if not selected_etfs:
        raise ValueError("At least one configured ticker must be selected")

    history_dir = Path(output_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    if not history_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {history_dir}")

    results: list[FundHistoryUpdateResult] = []
    errors: dict[str, str] = {}
    for etf in selected_etfs:
        try:
            migration = migrate_one_sector_etf_fund_history(
                etf,
                output_dir=history_dir,
            )
            LOGGER.info(
                "State Street %s fund-history path migration: %s",
                etf.ticker,
                migration.action,
            )
            result = update_one_sector_etf_fund_history(
                etf,
                config,
                output_dir=history_dir,
                http_client=http_client,
                max_attempts=max_attempts,
                sleep_func=sleep_func,
            )
            results.append(result)
            LOGGER.info(
                "Updated State Street %s history: rows=%s inserted=%s "
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
                "State Street fund history failed for %s: %s",
                etf.ticker,
                message,
            )

    if not results:
        LOGGER.warning(
            "All %s requested State Street fund-history updates failed; "
            "existing local files were preserved",
            len(selected_etfs),
        )

    mode = "bootstrap" if bootstrap else "full_refresh" if full_refresh else "daily"
    return SectorETFFundHistorySummary(
        configured_etfs=len(config.etfs),
        requested_etfs=len(selected_etfs),
        succeeded=len(results),
        failed=len(errors),
        files_written=sum(result.file_written for result in results),
        files_unchanged=sum(not result.file_written for result in results),
        rows_inserted=sum(result.inserted_rows for result in results),
        rows_updated=sum(result.updated_rows for result in results),
        consistency_warning_rows=sum(
            result.consistency_warning_rows for result in results
        ),
        severe_consistency_rows=sum(
            result.severe_consistency_rows for result in results
        ),
        zero_share_rows_normalized=sum(
            result.zero_share_rows_normalized for result in results
        ),
        mode=mode,
        errors=errors,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Update official State Street NAV, shares, and total-net-assets "
            "history for GICS sector ETFs."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--bootstrap", action="store_true")
    mode.add_argument("--full-refresh", action="store_true")
    parser.add_argument(
        "--tickers",
        help="Comma-separated configured tickers, case-insensitive.",
    )
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    tickers = (
        [ticker for ticker in args.tickers.split(",")]
        if args.tickers
        else None
    )
    try:
        summary = run_sector_etf_fund_history_update(
            tickers=tickers,
            bootstrap=args.bootstrap,
            full_refresh=args.full_refresh,
        )
    except Exception as error:
        print(
            f"State Street fund-history update failed: {short_error(error)}",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    print(summary.format())
    if summary.succeeded == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

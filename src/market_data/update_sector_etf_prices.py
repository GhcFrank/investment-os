"""
Download and maintain Yahoo Finance OHLCV history for GICS sector ETFs.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from io import StringIO
from typing import Any

import pandas as pd
import yfinance as yf

from src.market_data.sector_etf_config import (
    BASE_DIR,
    DEFAULT_CONFIG_FILE,
    SectorETF,
    SectorETFConfig,
    load_sector_etf_config,
)
from src.utils.csv_utils import atomic_write_csv
from src.utils.date_utils import MARKET_TIMEZONE
from src.utils.retry_utils import retry_call, short_error


LOGGER = logging.getLogger(__name__)

DEFAULT_PRICE_FILE = BASE_DIR / "data" / "market_data" / "sector_etf_prices.csv"
SOURCE_NAME = "yahoo_finance"

PRICE_COLUMNS = [
    "date",
    "ticker",
    "sector_id",
    "sector_name",
    "sector_name_cn",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "source",
    "fetched_at_utc",
]

YAHOO_PRICE_FIELDS = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "adj_close",
    "adjclose": "adj_close",
    "volume": "volume",
}


@dataclass(frozen=True)
class UpsertStats:
    inserted: int = 0
    updated: int = 0
    total: int = 0
    file_written: bool = False


@dataclass
class PriceDownloadResult:
    prices: pd.DataFrame
    succeeded: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SectorETFPriceUpdateSummary:
    configured_etfs: int
    price_tickers_succeeded: int = 0
    price_tickers_failed: int = 0
    price_rows_inserted: int = 0
    price_rows_updated: int = 0

    def format(self) -> str:
        return "\n".join(
            [
                "Sector ETF price update summary:",
                f"- configured ETFs: {self.configured_etfs}",
                (
                    "- price tickers succeeded: "
                    f"{self.price_tickers_succeeded}"
                ),
                f"- price tickers failed: {self.price_tickers_failed}",
                f"- price rows inserted: {self.price_rows_inserted}",
                f"- price rows updated: {self.price_rows_updated}",
            ]
        )


def now_utc_iso() -> str:
    return (
        datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _etf_sequence(
    config_or_etfs: SectorETFConfig | SectorETF | Sequence[SectorETF],
) -> tuple[SectorETF, ...]:
    if isinstance(config_or_etfs, SectorETFConfig):
        return config_or_etfs.leadership_etfs
    if isinstance(config_or_etfs, SectorETF):
        return (config_or_etfs,)
    return tuple(config_or_etfs)


def _field_name(column: object) -> str | None:
    values = column if isinstance(column, tuple) else (column,)
    for value in values:
        normalized = " ".join(str(value).strip().lower().split())
        if normalized in YAHOO_PRICE_FIELDS:
            return YAHOO_PRICE_FIELDS[normalized]
    return None


def _extract_ticker_frame(
    raw_prices: pd.DataFrame,
    ticker: str,
    allow_plain_columns: bool,
) -> pd.DataFrame:
    if not isinstance(raw_prices.columns, pd.MultiIndex):
        if not allow_plain_columns:
            raise ValueError(
                "Plain yfinance columns require a single ETF or an explicit "
                "ticker"
            )
        return raw_prices.copy()

    upper_ticker = ticker.upper()
    for level in range(raw_prices.columns.nlevels):
        matching_value = next(
            (
                value
                for value in raw_prices.columns.get_level_values(level)
                if str(value).strip().upper() == upper_ticker
            ),
            None,
        )
        if matching_value is not None:
            return raw_prices.xs(
                matching_value,
                axis=1,
                level=level,
                drop_level=True,
            ).copy()

    if allow_plain_columns:
        return raw_prices.copy()
    return pd.DataFrame(index=raw_prices.index)


def _normalize_market_date(value: object) -> str:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return ""
    if pd.isna(timestamp):
        return ""
    return timestamp.date().isoformat()


def _empty_prices() -> pd.DataFrame:
    return pd.DataFrame(columns=PRICE_COLUMNS)


def _clean_price_rows(prices: pd.DataFrame) -> pd.DataFrame:
    output = prices.copy()
    for column in PRICE_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA
    output = output.reindex(columns=PRICE_COLUMNS)
    output["date"] = output["date"].map(_normalize_market_date)
    output["ticker"] = (
        output["ticker"].fillna("").astype(str).str.strip().str.upper()
    )

    numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    for column in numeric_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output[numeric_columns] = output[numeric_columns].replace(
        [math.inf, -math.inf],
        pd.NA,
    )
    integral_volume = output["volume"].isna() | output["volume"].mod(1).eq(0)
    output.loc[~integral_volume, "volume"] = pd.NA
    output["volume"] = output["volume"].astype("Int64")

    output = output[
        output["date"].ne("")
        & output["ticker"].ne("")
        & output["close"].notna()
    ]
    output = output.drop_duplicates(
        subset=["date", "ticker"],
        keep="last",
    )
    output = output.sort_values(
        ["date", "ticker"],
        kind="stable",
    ).reset_index(drop=True)
    return output.reindex(columns=PRICE_COLUMNS)


def normalize_sector_etf_prices(
    raw_prices: pd.DataFrame,
    config_or_etfs: SectorETFConfig | SectorETF | Sequence[SectorETF],
    *,
    ticker: str | None = None,
    fetched_at_utc: str | None = None,
) -> pd.DataFrame:
    """
    Convert single- or multi-ticker yfinance output to the stable long schema.
    """

    if raw_prices is None or raw_prices.empty:
        return _empty_prices()

    etfs = _etf_sequence(config_or_etfs)
    if ticker is not None:
        selected_ticker = ticker.strip().upper()
        etfs = tuple(etf for etf in etfs if etf.ticker == selected_ticker)
        if not etfs:
            raise ValueError(f"Ticker is not configured: {selected_ticker}")
    if not etfs:
        return _empty_prices()

    fetched_at = fetched_at_utc or now_utc_iso()
    normalized_frames: list[pd.DataFrame] = []
    for etf in etfs:
        ticker_frame = _extract_ticker_frame(
            raw_prices,
            etf.ticker,
            ticker is not None or len(etfs) == 1,
        )
        if ticker_frame.empty:
            continue

        renamed = {
            column: canonical
            for column in ticker_frame.columns
            if (canonical := _field_name(column)) is not None
        }
        ticker_frame = ticker_frame.rename(columns=renamed)
        row_data: dict[str, Any] = {
            "date": [
                _normalize_market_date(value)
                for value in ticker_frame.index
            ],
            "ticker": etf.ticker,
            "sector_id": etf.sector_id,
            "sector_name": etf.sector_name,
            "sector_name_cn": etf.sector_name_cn,
            "source": SOURCE_NAME,
            "fetched_at_utc": fetched_at,
        }
        for column in ["open", "high", "low", "close", "adj_close", "volume"]:
            if column not in ticker_frame.columns:
                row_data[column] = pd.NA
                continue
            values = ticker_frame[column]
            if isinstance(values, pd.DataFrame):
                values = values.iloc[:, 0]
            row_data[column] = values.to_numpy()
        normalized_frames.append(pd.DataFrame(row_data))

    if not normalized_frames:
        return _empty_prices()
    return _clean_price_rows(
        pd.concat(normalized_frames, ignore_index=True)
    )


def _parse_iso_date(value: str, option_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            f"{option_name} must use YYYY-MM-DD format: {value}"
        ) from error


def _exclusive_end_date(end_date: str | None) -> str:
    inclusive_end = (
        _parse_iso_date(end_date, "end_date")
        if end_date
        else datetime.now(MARKET_TIMEZONE).date()
    )
    return (inclusive_end + timedelta(days=1)).isoformat()


def _latest_price_dates(prices: pd.DataFrame) -> dict[str, date]:
    if prices.empty:
        return {}
    latest: dict[str, date] = {}
    for ticker, rows in prices.groupby("ticker"):
        dates = pd.to_datetime(rows["date"], errors="coerce").dropna()
        if not dates.empty:
            latest[str(ticker).upper()] = dates.max().date()
    return latest


def download_sector_etf_prices(
    config: SectorETFConfig,
    *,
    latest_dates: Mapping[str, date] | None = None,
    full_refresh: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    fetched_at_utc: str | None = None,
    yf_module: Any | None = None,
    max_attempts: int = 3,
    sleep_func: Callable[[float], None] = time.sleep,
) -> PriceDownloadResult:
    """
    Download each ticker independently so one Yahoo failure stays isolated.
    """

    parsed_start = (
        _parse_iso_date(start_date, "start_date")
        if start_date
        else None
    )
    parsed_end = _parse_iso_date(end_date, "end_date") if end_date else None
    if parsed_start and parsed_end and parsed_start > parsed_end:
        raise ValueError("start_date cannot be later than end_date")
    if full_refresh and start_date:
        raise ValueError("full_refresh and start_date cannot be used together")

    yahoo = yf_module or yf
    latest_dates = {
        str(key).upper(): value
        for key, value in (latest_dates or {}).items()
    }
    exclusive_end = _exclusive_end_date(end_date)
    fetched_at = fetched_at_utc or now_utc_iso()
    frames: list[pd.DataFrame] = []
    succeeded: list[str] = []
    errors: dict[str, str] = {}

    for etf in config.leadership_etfs:
        kwargs: dict[str, Any] = {
            "tickers": etf.ticker,
            "end": exclusive_end,
            "interval": config.price_history.interval,
            "auto_adjust": False,
            "actions": False,
            "group_by": "ticker",
            "threads": False,
            "progress": False,
        }
        if parsed_start is not None:
            kwargs["start"] = parsed_start.isoformat()
        elif not full_refresh and etf.ticker in latest_dates:
            overlap_start = latest_dates[etf.ticker] - timedelta(
                days=config.price_history.incremental_overlap_days
            )
            kwargs["start"] = overlap_start.isoformat()
        else:
            kwargs["period"] = config.price_history.initial_period

        def fetch() -> pd.DataFrame:
            raw = yahoo.download(**kwargs)
            if raw is None or raw.empty:
                raise RuntimeError("Yahoo returned no price rows")
            return raw

        try:
            raw_prices = retry_call(
                fetch,
                label=f"Yahoo prices for {etf.ticker}",
                max_attempts=max_attempts,
                sleep_func=sleep_func,
                logger=LOGGER,
            )
            normalized = normalize_sector_etf_prices(
                raw_prices,
                etf,
                ticker=etf.ticker,
                fetched_at_utc=fetched_at,
            )
            if normalized.empty:
                raise RuntimeError(
                    "Yahoo price rows were invalid after normalization"
                )
            frames.append(normalized)
            succeeded.append(etf.ticker)
        except Exception as error:
            message = short_error(error)
            errors[etf.ticker] = message
            LOGGER.error("Yahoo prices failed for %s: %s", etf.ticker, message)

    prices = (
        _clean_price_rows(pd.concat(frames, ignore_index=True))
        if frames
        else _empty_prices()
    )
    return PriceDownloadResult(
        prices=prices,
        succeeded=succeeded,
        errors=errors,
    )


def _read_existing_prices(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _empty_prices()
    try:
        existing = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return _empty_prices()
    if list(existing.columns) != PRICE_COLUMNS:
        raise ValueError(
            f"CSV schema mismatch for {path}: expected {PRICE_COLUMNS}, "
            f"found {list(existing.columns)}"
        )
    return _clean_price_rows(existing)


def upsert_sector_etf_prices(
    new_prices: pd.DataFrame,
    output_file: Path | str = DEFAULT_PRICE_FILE,
) -> UpsertStats:
    """
    Insert new date/ticker rows and replace overlapping rows atomically.
    """

    path = Path(output_file)
    existing = _read_existing_prices(path)
    incoming = _clean_price_rows(new_prices)
    existing_keys = set(zip(existing["date"], existing["ticker"]))
    incoming_keys = set(zip(incoming["date"], incoming["ticker"]))
    comparison_columns = [
        column
        for column in PRICE_COLUMNS
        if column not in {"date", "ticker", "fetched_at_utc"}
    ]
    existing_rows = {
        (str(row.date), str(row.ticker)): row
        for row in existing.itertuples(index=False)
    }
    updated_keys: set[tuple[str, str]] = set()
    for row_index, row in incoming.iterrows():
        key = (str(row["date"]), str(row["ticker"]))
        existing_row = existing_rows.get(key)
        if existing_row is None:
            continue
        unchanged = all(
            (
                pd.isna(row[column])
                and pd.isna(getattr(existing_row, column))
            )
            or (
                not pd.isna(row[column])
                and not pd.isna(getattr(existing_row, column))
                and row[column] == getattr(existing_row, column)
            )
            for column in comparison_columns
        )
        if unchanged:
            incoming.at[row_index, "fetched_at_utc"] = (
                existing_row.fetched_at_utc
            )
        else:
            updated_keys.add(key)
    combined = _clean_price_rows(
        pd.concat([existing, incoming], ignore_index=True)
    )

    buffer = StringIO()
    combined.to_csv(
        buffer,
        columns=PRICE_COLUMNS,
        index=False,
        encoding="utf-8",
        na_rep="",
        lineterminator="\n",
    )
    expected_content = buffer.getvalue()
    file_written = (
        not path.exists()
        or path.read_text(encoding="utf-8") != expected_content
    )
    if file_written:
        atomic_write_csv(
            combined,
            path,
            PRICE_COLUMNS,
            na_rep="",
            lineterminator="\n",
        )
    return UpsertStats(
        inserted=len(incoming_keys - existing_keys),
        updated=len(updated_keys),
        total=len(combined),
        file_written=file_written,
    )


def run_sector_etf_price_update(
    *,
    config_path: Path | str = DEFAULT_CONFIG_FILE,
    price_file: Path | str = DEFAULT_PRICE_FILE,
    full_refresh: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
) -> SectorETFPriceUpdateSummary:
    """
    Update Yahoo OHLCV data without requesting ETF fund metadata.
    """

    if full_refresh and start_date:
        raise ValueError("full_refresh and start_date cannot be used together")
    config = load_sector_etf_config(config_path)
    existing_prices = _read_existing_prices(Path(price_file))
    result = download_sector_etf_prices(
        config,
        latest_dates=_latest_price_dates(existing_prices),
        full_refresh=full_refresh,
        start_date=start_date,
        end_date=end_date,
    )
    stats = UpsertStats(total=len(existing_prices))
    if not result.prices.empty:
        stats = upsert_sector_etf_prices(result.prices, price_file)

    summary = SectorETFPriceUpdateSummary(
        configured_etfs=len(config.leadership_etfs),
        price_tickers_succeeded=len(result.succeeded),
        price_tickers_failed=len(result.errors),
        price_rows_inserted=stats.inserted,
        price_rows_updated=stats.updated,
    )
    if result.prices.empty:
        raise RuntimeError(
            "No valid sector ETF price data was downloaded; existing price "
            "history was left unchanged. "
            + summary.format().replace("\n", " ")
        )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update Yahoo OHLCV history for GICS sector ETFs."
    )
    parser.add_argument("--full-refresh", action="store_true")
    parser.add_argument("--start-date")
    parser.add_argument(
        "--end-date",
        help="Inclusive market date in YYYY-MM-DD format.",
    )
    args = parser.parse_args(argv)
    if args.full_refresh and args.start_date:
        parser.error(
            "--full-refresh and --start-date cannot be used together"
        )
    try:
        if args.start_date:
            _parse_iso_date(args.start_date, "--start-date")
        if args.end_date:
            _parse_iso_date(args.end_date, "--end-date")
    except ValueError as error:
        parser.error(str(error))
    if (
        args.start_date
        and args.end_date
        and args.start_date > args.end_date
    ):
        parser.error("--start-date cannot be later than --end-date")
    return args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    try:
        summary = run_sector_etf_price_update(
            full_refresh=args.full_refresh,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    except Exception as error:
        print(
            f"Sector ETF price update failed: {short_error(error)}",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    print(summary.format())


if __name__ == "__main__":
    main()

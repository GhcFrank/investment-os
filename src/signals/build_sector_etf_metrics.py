"""
Build auditable trading-day adjusted-close returns for sector ETFs.

The module is intentionally local-only: it reads the Yahoo price history that
the market-data step has already written and never downloads data itself.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

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
from src.utils.retry_utils import short_error


LOGGER = logging.getLogger(__name__)

DEFAULT_PRICE_FILE = BASE_DIR / "data" / "market_data" / "sector_etf_prices.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "signals" / "sector_etf_metrics"
SECTOR_ETF_RETURN_HORIZONS = (250, 90, 30)
METRICS_FLOAT_FORMAT = "%.15g"
REQUIRED_PRICE_COLUMNS = ["date", "ticker", "adj_close"]
METRICS_COLUMNS = [
    "date",
    "adj_close",
    "reference_date_250td",
    "reference_adj_close_250td",
    "adj_close_return_250td",
    "reference_date_90td",
    "reference_adj_close_90td",
    "adj_close_return_90td",
    "reference_date_30td",
    "reference_adj_close_30td",
    "adj_close_return_30td",
]


class SectorETFMetricsValidationError(ValueError):
    """Sector ETF price input or metrics output is structurally invalid."""


@dataclass(frozen=True)
class SectorETFMetricsResult:
    ticker: str
    output_path: Path
    rows: int
    earliest_date: str
    latest_date: str
    return_250td_non_null: int
    return_90td_non_null: int
    return_30td_non_null: int
    file_written: bool


@dataclass(frozen=True)
class SectorETFMetricsUpdateSummary:
    configured_etfs: int
    succeeded: int
    failed: int
    files_written: int
    files_unchanged: int
    results: tuple[SectorETFMetricsResult, ...] = ()
    errors: dict[str, str] = field(default_factory=dict)

    def format(self) -> str:
        lines = [
            "Sector ETF adjusted-close metrics update summary:",
            f"- configured ETFs: {self.configured_etfs}",
            f"- succeeded: {self.succeeded}",
            f"- failed: {self.failed}",
            f"- files written: {self.files_written}",
            f"- files unchanged: {self.files_unchanged}",
        ]
        for result in self.results:
            lines.append(
                f"- {result.ticker}: rows={result.rows}, "
                f"range={result.earliest_date}..{result.latest_date}, "
                f"250td={result.return_250td_non_null}, "
                f"90td={result.return_90td_non_null}, "
                f"30td={result.return_30td_non_null}, "
                f"written={result.file_written}, "
                f"file={result.output_path.name}"
            )
        for ticker, message in self.errors.items():
            lines.append(f"- {ticker} error: {message}")
        return "\n".join(lines)


def _parse_price_dates(values: pd.Series, *, context: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    invalid = parsed.isna()
    if invalid.any():
        examples = values.loc[invalid].astype(str).head(3).tolist()
        raise SectorETFMetricsValidationError(
            f"{context} contains invalid date value(s): {examples}"
        )
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_convert(MARKET_TIMEZONE).dt.tz_localize(None)
    return parsed.dt.normalize()


def validate_sector_etf_prices(
    prices: pd.DataFrame,
    *,
    expected_tickers: Collection[str] | None = None,
    today: date | None = None,
) -> pd.DataFrame:
    """
    Validate and normalize the local long-format adjusted-close history.
    """

    if not isinstance(prices, pd.DataFrame):
        raise TypeError("Sector ETF prices must be a pandas DataFrame")
    missing_columns = [
        column for column in REQUIRED_PRICE_COLUMNS if column not in prices
    ]
    if missing_columns:
        raise SectorETFMetricsValidationError(
            "sector_etf_prices.csv missing required column(s): "
            + ", ".join(missing_columns)
            + "; adjusted-close metrics never fall back to raw close"
        )
    if prices.empty:
        raise SectorETFMetricsValidationError(
            "sector_etf_prices.csv contains no price rows"
        )

    output = prices.loc[:, REQUIRED_PRICE_COLUMNS].copy()
    output["date"] = _parse_price_dates(
        output["date"],
        context="sector_etf_prices.csv",
    )

    normalized_tickers = (
        output["ticker"].fillna("").astype(str).str.strip().str.upper()
    )
    if normalized_tickers.eq("").any():
        raise SectorETFMetricsValidationError(
            "sector_etf_prices.csv contains an empty ticker"
        )
    output["ticker"] = normalized_tickers

    numeric_adj_close = pd.to_numeric(output["adj_close"], errors="coerce")
    numeric_values = numeric_adj_close.to_numpy(dtype=float, na_value=np.nan)
    if numeric_adj_close.isna().any():
        invalid_values = (
            prices.loc[numeric_adj_close.isna(), "adj_close"]
            .astype(str)
            .head(3)
            .tolist()
        )
        raise SectorETFMetricsValidationError(
            "sector_etf_prices.csv contains missing or non-numeric adj_close "
            f"value(s): {invalid_values}"
        )
    if not np.isfinite(numeric_values).all():
        raise SectorETFMetricsValidationError(
            "sector_etf_prices.csv contains infinite adj_close values"
        )
    if (numeric_adj_close <= 0).any():
        raise SectorETFMetricsValidationError(
            "sector_etf_prices.csv adj_close values must be greater than zero"
        )
    output["adj_close"] = numeric_adj_close.astype(float)

    duplicate_rows = output.duplicated(["date", "ticker"], keep=False)
    if duplicate_rows.any():
        duplicate_keys = (
            output.loc[duplicate_rows, ["date", "ticker"]]
            .head(3)
            .assign(date=lambda frame: frame["date"].dt.strftime("%Y-%m-%d"))
            .to_dict("records")
        )
        raise SectorETFMetricsValidationError(
            "sector_etf_prices.csv contains duplicate date + ticker rows: "
            f"{duplicate_keys}"
        )

    market_today = today or datetime.now(MARKET_TIMEZONE).date()
    future_rows = output["date"].dt.date > market_today
    if future_rows.any():
        future_dates = (
            output.loc[future_rows, "date"]
            .dt.strftime("%Y-%m-%d")
            .head(3)
            .tolist()
        )
        raise SectorETFMetricsValidationError(
            "sector_etf_prices.csv contains future market date(s) relative "
            f"to America/New_York {market_today.isoformat()}: {future_dates}"
        )

    if expected_tickers is not None:
        expected = {
            str(ticker).strip().upper()
            for ticker in expected_tickers
            if str(ticker).strip()
        }
        actual = set(output["ticker"])
        missing_tickers = sorted(expected - actual)
        unexpected_tickers = sorted(actual - expected)
        problems: list[str] = []
        if missing_tickers:
            problems.append(
                "missing configured ticker(s): " + ", ".join(missing_tickers)
            )
        if unexpected_tickers:
            problems.append(
                "unconfigured ticker(s): " + ", ".join(unexpected_tickers)
            )
        if problems:
            raise SectorETFMetricsValidationError(
                "sector_etf_prices.csv " + "; ".join(problems)
            )

    return output.sort_values(
        ["ticker", "date"],
        kind="stable",
    ).reset_index(drop=True)


def load_sector_etf_prices(
    price_file: Path | str = DEFAULT_PRICE_FILE,
    *,
    expected_tickers: Collection[str] | None = None,
    today: date | None = None,
) -> pd.DataFrame:
    """Read and validate the local price CSV without any network access."""

    path = Path(price_file)
    if not path.exists():
        raise FileNotFoundError(f"Sector ETF price file not found: {path}")
    try:
        prices = pd.read_csv(path)
    except pd.errors.EmptyDataError as error:
        raise SectorETFMetricsValidationError(
            f"Sector ETF price file is empty: {path}"
        ) from error
    return validate_sector_etf_prices(
        prices,
        expected_tickers=expected_tickers,
        today=today,
    )


def calculate_trading_day_adj_close_return(
    prices: pd.DataFrame,
    horizon_trading_days: int,
) -> pd.DataFrame:
    """
    Calculate one return horizon from the ETF's own observed price rows.

    Row ``i`` uses row ``i - horizon_trading_days`` as its reference. No
    calendar, business-day approximation, or other ticker's observations are
    involved.
    """

    if (
        isinstance(horizon_trading_days, bool)
        or not isinstance(horizon_trading_days, int)
        or horizon_trading_days <= 0
    ):
        raise ValueError(
            "horizon_trading_days must be a positive integer"
        )
    missing_columns = [
        column for column in ("date", "adj_close") if column not in prices
    ]
    if missing_columns:
        raise SectorETFMetricsValidationError(
            "Trading-day return input missing column(s): "
            + ", ".join(missing_columns)
        )

    if "ticker" in prices:
        tickers = (
            prices["ticker"].fillna("").astype(str).str.strip().str.upper()
        )
        if tickers.eq("").any() or tickers.nunique() != 1:
            raise SectorETFMetricsValidationError(
                "Trading-day return input must contain exactly one ticker"
            )

    history = prices.loc[:, ["date", "adj_close"]].copy()
    history["date"] = _parse_price_dates(
        history["date"],
        context="trading-day return input",
    )
    history["adj_close"] = pd.to_numeric(
        history["adj_close"],
        errors="coerce",
    ).astype(float)
    price_values = history["adj_close"].to_numpy(
        dtype=float,
        na_value=np.nan,
    )
    if (
        history["adj_close"].isna().any()
        or not np.isfinite(price_values).all()
        or (history["adj_close"] <= 0).any()
    ):
        raise SectorETFMetricsValidationError(
            "Trading-day return input adj_close values must be finite and "
            "positive"
        )
    history = history.sort_values("date", kind="stable").reset_index(drop=True)
    if history["date"].duplicated().any():
        raise SectorETFMetricsValidationError(
            "Trading-day return input contains duplicate dates"
        )

    suffix = f"{horizon_trading_days}td"
    reference_date_column = f"reference_date_{suffix}"
    reference_price_column = f"reference_adj_close_{suffix}"
    return_column = f"adj_close_return_{suffix}"
    reference_dates = history["date"].shift(horizon_trading_days)
    reference_prices = history["adj_close"].shift(horizon_trading_days)
    returns = history["adj_close"] / reference_prices - 1.0
    return pd.DataFrame(
        {
            reference_date_column: reference_dates,
            reference_price_column: reference_prices,
            return_column: returns,
        }
    )


def build_one_sector_etf_metrics(prices: pd.DataFrame) -> pd.DataFrame:
    """Build all configured trading-day horizons for one ETF."""

    normalized = validate_sector_etf_prices(prices)
    tickers = normalized["ticker"].unique()
    if len(tickers) != 1:
        raise SectorETFMetricsValidationError(
            "One-ETF metrics input must contain exactly one ticker; found "
            + ", ".join(sorted(tickers))
        )
    normalized = normalized.sort_values("date", kind="stable").reset_index(
        drop=True
    )
    output = normalized.loc[:, ["date", "adj_close"]].copy()
    for horizon_trading_days in SECTOR_ETF_RETURN_HORIZONS:
        horizon_metrics = calculate_trading_day_adj_close_return(
            normalized,
            horizon_trading_days,
        )
        output = pd.concat([output, horizon_metrics], axis=1)

    output["date"] = output["date"].dt.strftime("%Y-%m-%d")
    for horizon_trading_days in SECTOR_ETF_RETURN_HORIZONS:
        reference_column = (
            f"reference_date_{horizon_trading_days}td"
        )
        output[reference_column] = output[reference_column].dt.strftime(
            "%Y-%m-%d"
        )
    output = output.reindex(columns=METRICS_COLUMNS)
    validate_sector_etf_metrics(output)
    return output


def build_all_sector_etf_metrics(
    prices: pd.DataFrame,
    config: SectorETFConfig,
) -> dict[str, pd.DataFrame]:
    """Build independent metric frames for every configured ETF."""

    normalized = validate_sector_etf_prices(
        prices,
        expected_tickers={etf.ticker for etf in config.leadership_etfs},
    )
    return {
        etf.ticker: build_one_sector_etf_metrics(
            normalized.loc[normalized["ticker"].eq(etf.ticker)]
        )
        for etf in config.leadership_etfs
    }


def validate_sector_etf_metrics(metrics: pd.DataFrame) -> None:
    """Validate schema, chronology, reference integrity, and return formulas."""

    if list(metrics.columns) != METRICS_COLUMNS:
        raise SectorETFMetricsValidationError(
            f"Metrics schema mismatch: expected {METRICS_COLUMNS}, "
            f"found {list(metrics.columns)}"
        )
    if metrics.empty:
        raise SectorETFMetricsValidationError("Metrics output cannot be empty")

    dates = _parse_price_dates(metrics["date"], context="metrics output")
    if dates.duplicated().any():
        raise SectorETFMetricsValidationError(
            "Metrics output contains duplicate dates"
        )
    if not dates.is_monotonic_increasing:
        raise SectorETFMetricsValidationError(
            "Metrics output dates must be ascending"
        )

    current_prices = pd.to_numeric(metrics["adj_close"], errors="coerce")
    current_values = current_prices.to_numpy(dtype=float, na_value=np.nan)
    if (
        current_prices.isna().any()
        or not np.isfinite(current_values).all()
        or (current_prices <= 0).any()
    ):
        raise SectorETFMetricsValidationError(
            "Metrics output adj_close values must be finite and positive"
        )

    for horizon_trading_days in SECTOR_ETF_RETURN_HORIZONS:
        suffix = f"{horizon_trading_days}td"
        reference_date_column = f"reference_date_{suffix}"
        reference_price_column = f"reference_adj_close_{suffix}"
        return_column = f"adj_close_return_{suffix}"

        raw_reference_dates = metrics[reference_date_column]
        blank_reference_dates = raw_reference_dates.isna() | (
            raw_reference_dates.astype(str).str.strip().eq("")
        )
        reference_dates = pd.to_datetime(
            raw_reference_dates,
            errors="coerce",
            format="mixed",
        )
        invalid_reference_dates = (
            reference_dates.isna() & ~blank_reference_dates
        )
        if invalid_reference_dates.any():
            raise SectorETFMetricsValidationError(
                f"Metrics output has invalid {reference_date_column}"
            )

        reference_prices = pd.to_numeric(
            metrics[reference_price_column],
            errors="coerce",
        )
        returns = pd.to_numeric(metrics[return_column], errors="coerce")
        missingness_matches = (
            blank_reference_dates
            .eq(reference_prices.isna())
            .all()
            and blank_reference_dates.eq(returns.isna()).all()
        )
        if not missingness_matches:
            raise SectorETFMetricsValidationError(
                f"Metrics output {suffix} reference and return nulls "
                "must align"
            )

        available = ~blank_reference_dates
        expected_available = pd.Series(
            np.arange(len(metrics)) >= horizon_trading_days,
            index=metrics.index,
        )
        if not available.equals(expected_available):
            raise SectorETFMetricsValidationError(
                f"Metrics output {suffix} must be empty for exactly the "
                f"first {horizon_trading_days} trading observations"
            )
        if not expected_available.any():
            continue
        reference_values = reference_prices.loc[available].to_numpy(
            dtype=float,
            na_value=np.nan,
        )
        return_values = returns.loc[available].to_numpy(
            dtype=float,
            na_value=np.nan,
        )
        if (
            not np.isfinite(reference_values).all()
            or (reference_values <= 0).any()
            or not np.isfinite(return_values).all()
        ):
            raise SectorETFMetricsValidationError(
                f"Metrics output {suffix} values must be finite and "
                "reference prices must be positive"
            )

        expected_reference_dates = dates.shift(horizon_trading_days)
        if not np.array_equal(
            reference_dates.loc[available].to_numpy(
                dtype="datetime64[ns]"
            ),
            expected_reference_dates.loc[available].to_numpy(
                dtype="datetime64[ns]"
            ),
        ):
            raise SectorETFMetricsValidationError(
                f"Metrics output {suffix} reference dates must be exactly "
                f"{horizon_trading_days} trading observations earlier"
            )

        expected_reference_prices = current_prices.shift(
            horizon_trading_days
        )
        if not np.allclose(
            reference_values,
            expected_reference_prices.loc[available].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise SectorETFMetricsValidationError(
                f"Metrics output {suffix} reference prices must come from "
                f"exactly {horizon_trading_days} trading observations earlier"
            )

        expected_returns = (
            current_prices.loc[available].to_numpy(dtype=float)
            / reference_values
            - 1.0
        )
        if not np.allclose(
            return_values,
            expected_returns,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise SectorETFMetricsValidationError(
                f"Metrics output {suffix} return formula is invalid"
            )


def resolve_sector_etf_metrics_path(
    output_dir: Path | str,
    etf: SectorETF,
) -> Path:
    """Resolve the configured sector basename inside the metrics directory."""

    filename = validate_fund_history_filename(
        etf.metrics_filename,
        location=f"{etf.ticker}.metrics_filename",
    )
    return Path(output_dir) / filename


def _canonical_metrics_csv(metrics: pd.DataFrame) -> str:
    validate_sector_etf_metrics(metrics)
    buffer = StringIO()
    metrics.to_csv(
        buffer,
        columns=METRICS_COLUMNS,
        index=False,
        encoding="utf-8",
        float_format=METRICS_FLOAT_FORMAT,
        na_rep="",
        lineterminator="\n",
    )
    return buffer.getvalue()


def write_metrics_atomic(
    metrics: pd.DataFrame,
    path: Path | str,
) -> bool:
    """
    Atomically write canonical CSV content only when bytes have changed.

    Returns ``True`` when the target was replaced and ``False`` when the
    existing canonical file was already identical.
    """

    output_path = Path(path)
    expected_content = _canonical_metrics_csv(metrics)
    if (
        output_path.exists()
        and output_path.read_text(encoding="utf-8") == expected_content
    ):
        return False

    def verify_temp_file(temp_path: Path) -> None:
        if temp_path.read_text(encoding="utf-8") != expected_content:
            raise SectorETFMetricsValidationError(
                f"Temporary metrics CSV verification failed: {temp_path}"
            )

    atomic_write_csv(
        metrics,
        output_path,
        METRICS_COLUMNS,
        float_format=METRICS_FLOAT_FORMAT,
        na_rep="",
        lineterminator="\n",
        validate_temp_file=verify_temp_file,
    )
    return True


def _result_for_metrics(
    etf: SectorETF,
    output_path: Path,
    metrics: pd.DataFrame,
    *,
    file_written: bool,
) -> SectorETFMetricsResult:
    return SectorETFMetricsResult(
        ticker=etf.ticker,
        output_path=output_path,
        rows=len(metrics),
        earliest_date=str(metrics["date"].iloc[0]),
        latest_date=str(metrics["date"].iloc[-1]),
        return_250td_non_null=int(
            metrics["adj_close_return_250td"].notna().sum()
        ),
        return_90td_non_null=int(
            metrics["adj_close_return_90td"].notna().sum()
        ),
        return_30td_non_null=int(
            metrics["adj_close_return_30td"].notna().sum()
        ),
        file_written=file_written,
    )


def run_sector_etf_metrics_update(
    *,
    config_path: Path | str = DEFAULT_CONFIG_FILE,
    price_file: Path | str = DEFAULT_PRICE_FILE,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> SectorETFMetricsUpdateSummary:
    """
    Rebuild all history locally, isolating per-ETF calculation/write failures.
    """

    config = load_sector_etf_config(config_path)
    prices = load_sector_etf_prices(
        price_file,
        expected_tickers={etf.ticker for etf in config.leadership_etfs},
    )
    metrics_dir = Path(output_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if not metrics_dir.is_dir():
        raise NotADirectoryError(
            f"Metrics output path is not a directory: {metrics_dir}"
        )

    results: list[SectorETFMetricsResult] = []
    errors: dict[str, str] = {}
    for etf in config.leadership_etfs:
        try:
            etf_prices = prices.loc[prices["ticker"].eq(etf.ticker)]
            metrics = build_one_sector_etf_metrics(etf_prices)
            output_path = resolve_sector_etf_metrics_path(metrics_dir, etf)
            file_written = write_metrics_atomic(metrics, output_path)
            result = _result_for_metrics(
                etf,
                output_path,
                metrics,
                file_written=file_written,
            )
            results.append(result)
            LOGGER.info(
                "Built %s adjusted-close metrics: rows=%s written=%s",
                etf.ticker,
                result.rows,
                result.file_written,
            )
        except Exception as error:
            message = short_error(error)
            errors[etf.ticker] = message
            LOGGER.error(
                "Sector ETF metrics failed for %s: %s",
                etf.ticker,
                message,
            )

    return SectorETFMetricsUpdateSummary(
        configured_etfs=len(config.leadership_etfs),
        succeeded=len(results),
        failed=len(errors),
        files_written=sum(result.file_written for result in results),
        files_unchanged=sum(not result.file_written for result in results),
        results=tuple(results),
        errors=errors,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild local 30/90/250-trading-day sector ETF "
            "adjusted-close returns."
        )
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Explicitly request the default full-history rebuild. The default "
            "command also rebuilds all history."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    parse_args()
    try:
        summary = run_sector_etf_metrics_update()
    except Exception as error:
        print(
            f"Sector ETF metrics update failed: {short_error(error)}",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    print(summary.format())
    if summary.succeeded == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

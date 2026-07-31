"""
Build, persist, and optionally email daily sector ETF return rankings.

The module consumes the local per-ETF metrics files. It never downloads market
or fund data, and it never substitutes a prior date for a missing ETF.
"""

from __future__ import annotations

import argparse
import html
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from market_data.sector_etf_config import (
    BASE_DIR,
    DEFAULT_CONFIG_FILE,
    SectorETFConfig,
    load_sector_etf_config,
)
from signals.build_sector_etf_metrics import (
    DEFAULT_OUTPUT_DIR as DEFAULT_METRICS_DIR,
    METRICS_COLUMNS,
    METRICS_FLOAT_FORMAT,
    SECTOR_ETF_RETURN_HORIZONS,
    resolve_sector_etf_metrics_path,
    validate_sector_etf_metrics,
)
from utils.csv_utils import atomic_write_csv
from utils.retry_utils import short_error
from utils.send_email import send_email


LOGGER = logging.getLogger(__name__)

DEFAULT_RANKING_HISTORY_FILE = (
    BASE_DIR / "data" / "signals" / "sector_etf_daily_rankings.csv"
)
DEFAULT_EMAIL_LOG_FILE = (
    BASE_DIR / "data" / "signals" / "sector_etf_ranking_email_log.csv"
)
MIN_RANKING_UNIVERSE_SIZE = 6
RANKING_GROUP_ORDER = ("top", "bottom")
# Presentation-only order for the email body. Ranking calculation and
# persistence continue to use SECTOR_ETF_RETURN_HORIZONS.
EMAIL_HORIZON_ORDER = (30, 90, 250)
RANKING_COLUMNS = [
    "date",
    "horizon_trading_days",
    "ranking_group",
    "rank",
    "ticker",
    "sector_id",
    "sector_name",
    "sector_name_cn",
    "adj_close",
    "reference_date",
    "reference_adj_close",
    "adj_close_return",
    "universe_size",
]
EMAIL_LOG_COLUMNS = [
    "ranking_date",
    "sent_at_utc",
    "status",
    "recipient_count",
    "error_message",
]
TEST_EMAIL_BANNER_PLAIN = (
    "TEST EMAIL — Format validation only.\n"
    "This message does not represent a new daily production alert."
)
TEST_EMAIL_BANNER_HTML = (
    '<div style="border:2px solid #b45309;background:#fffbeb;'
    'padding:12px;margin-bottom:18px">'
    "<strong>TEST EMAIL — Format validation only.</strong><br>"
    "This message does not represent a new daily production alert."
    "</div>"
)


class SectorETFRankingValidationError(ValueError):
    """Ranking inputs or persisted ranking data are invalid."""


class InsufficientRankingUniverseError(SectorETFRankingValidationError):
    """Fewer than six ETFs have valid values for at least one horizon."""


@dataclass(frozen=True)
class LatestSectorETFMetrics:
    ranking_date: str
    rows: pd.DataFrame
    missing_tickers: tuple[str, ...]
    configured_count: int
    latest_dates: dict[str, str]

    @property
    def participating_count(self) -> int:
        return int(self.rows["ticker"].nunique())

    @property
    def is_complete(self) -> bool:
        return (
            self.participating_count == self.configured_count
            and not self.missing_tickers
        )


@dataclass(frozen=True)
class SectorETFRankingEmail:
    subject: str
    plain_text: str
    html: str
    incomplete: bool


@dataclass(frozen=True)
class RankingEmailSendResult:
    status: str
    recipient_count: int = 0
    error_message: str = ""


@dataclass(frozen=True)
class SectorETFTestEmailResult:
    ranking_date: str
    configured_etfs: int
    participating_etfs: int
    missing_tickers: tuple[str, ...]
    recipient_count: int
    email: SectorETFRankingEmail

    def format(self) -> str:
        lines = [
            "Sector ETF test email summary:",
            f"- ranking date: {self.ranking_date}",
            (
                "- participating ETFs: "
                f"{self.participating_etfs}/{self.configured_etfs}"
            ),
            f"- subject: {self.email.subject}",
            f"- recipient count: {self.recipient_count}",
            "- production email log updated: no",
        ]
        if self.missing_tickers:
            lines.append(
                "- missing tickers: " + ", ".join(self.missing_tickers)
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class SectorETFRankingSummary:
    ranking_date: str
    configured_etfs: int
    participating_etfs: int
    missing_tickers: tuple[str, ...]
    ranking_rows: int
    history_written: bool
    email_status: str
    email_error: str
    rankings: pd.DataFrame
    email: SectorETFRankingEmail

    def format(self) -> str:
        completeness = (
            "complete"
            if self.participating_etfs == self.configured_etfs
            and not self.missing_tickers
            else "incomplete"
        )
        lines = [
            "Sector ETF daily ranking summary:",
            f"- ranking date: {self.ranking_date}",
            (
                "- participating ETFs: "
                f"{self.participating_etfs}/{self.configured_etfs}"
            ),
            f"- data completeness: {completeness}",
            f"- ranking rows: {self.ranking_rows}",
            f"- ranking history written: {self.history_written}",
            f"- email status: {self.email_status}",
        ]
        if self.missing_tickers:
            lines.append(
                "- missing tickers: " + ", ".join(self.missing_tickers)
            )
        if self.email_error:
            lines.append(f"- email error: {self.email_error}")
        for horizon_trading_days in SECTOR_ETF_RETURN_HORIZONS:
            for ranking_group in RANKING_GROUP_ORDER:
                selected = self.rankings.loc[
                    self.rankings["horizon_trading_days"].eq(
                        horizon_trading_days
                    )
                    & self.rankings["ranking_group"].eq(ranking_group)
                ].sort_values("rank")
                values = ", ".join(
                    f"{row.ticker}={row.adj_close_return:.4%}"
                    for row in selected.itertuples()
                )
                lines.append(
                    f"- {horizon_trading_days}td {ranking_group}: {values}"
                )
        return "\n".join(lines)


@dataclass(frozen=True)
class SectorETFRankingHistoryRebuildSummary:
    configured_etfs: int
    common_dates: int
    ranked_dates: int
    skipped_unrankable_dates: int
    ranking_rows: int
    earliest_date: str
    latest_date: str
    history_written: bool

    def format(self) -> str:
        return "\n".join(
            [
                "Sector ETF ranking history rebuild summary:",
                f"- configured ETFs: {self.configured_etfs}",
                f"- common trading dates: {self.common_dates}",
                f"- ranked dates: {self.ranked_dates}",
                (
                    "- skipped dates without six valid ETFs in every "
                    f"horizon: {self.skipped_unrankable_dates}"
                ),
                f"- ranking rows: {self.ranking_rows}",
                f"- date range: {self.earliest_date}..{self.latest_date}",
                f"- ranking history written: {self.history_written}",
                "- email requested: no",
                "- production email log updated: no",
            ]
        )


def _parse_ranking_date(value: str, *, option_name: str) -> str:
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as error:
        raise SectorETFRankingValidationError(
            f"{option_name} must use YYYY-MM-DD format: {value}"
        ) from error


def _load_one_metrics_file(path: Path, ticker: str) -> pd.DataFrame:
    try:
        metrics = pd.read_csv(path)
    except pd.errors.EmptyDataError as error:
        raise SectorETFRankingValidationError(
            f"Metrics file is empty for {ticker}: {path}"
        ) from error
    if list(metrics.columns) != METRICS_COLUMNS:
        raise SectorETFRankingValidationError(
            f"Metrics schema mismatch for {ticker}: expected "
            f"{METRICS_COLUMNS}, found {list(metrics.columns)}"
        )
    validate_sector_etf_metrics(metrics)
    return metrics


def load_latest_sector_etf_metrics(
    config: SectorETFConfig,
    *,
    metrics_dir: Path | str = DEFAULT_METRICS_DIR,
    ranking_date: str | None = None,
) -> LatestSectorETFMetrics:
    """
    Load one exact date from each configured metrics file.

    The default date is the latest date present in any configured file. An ETF
    without that exact date is marked missing; its prior row is never used.
    """

    requested_date = (
        _parse_ranking_date(ranking_date, option_name="ranking_date")
        if ranking_date is not None
        else None
    )
    loaded: dict[str, pd.DataFrame] = {}
    latest_dates: dict[str, str] = {}
    missing_files: list[str] = []
    for etf in config.leadership_etfs:
        path = resolve_sector_etf_metrics_path(metrics_dir, etf)
        if not path.exists():
            missing_files.append(etf.ticker)
            continue
        etf_metrics = _load_one_metrics_file(path, etf.ticker)
        loaded[etf.ticker] = etf_metrics
        latest_dates[etf.ticker] = str(etf_metrics["date"].iloc[-1])

    if not loaded:
        raise FileNotFoundError(
            f"No configured sector ETF metrics files found in {metrics_dir}"
        )

    available_dates = {
        str(value)
        for frame in loaded.values()
        for value in frame["date"].astype(str)
    }
    if requested_date is not None:
        if requested_date not in available_dates:
            raise SectorETFRankingValidationError(
                f"Requested ranking date is not present in metrics: "
                f"{requested_date}"
            )
        selected_date = requested_date
    else:
        selected_date = max(latest_dates.values())

    rows: list[dict[str, object]] = []
    missing_tickers = list(missing_files)
    for etf in config.leadership_etfs:
        frame = loaded.get(etf.ticker)
        if frame is None:
            continue
        selected = frame.loc[frame["date"].astype(str).eq(selected_date)]
        if selected.empty:
            missing_tickers.append(etf.ticker)
            continue
        if len(selected) != 1:
            raise SectorETFRankingValidationError(
                f"{etf.ticker} has duplicate metrics rows for {selected_date}"
            )
        metric_row = selected.iloc[0].to_dict()
        rows.append(
            {
                **metric_row,
                "ticker": etf.ticker,
                "sector_id": etf.sector_id,
                "sector_name": etf.sector_name,
                "sector_name_cn": etf.sector_name_cn,
            }
        )

    latest = LatestSectorETFMetrics(
        ranking_date=selected_date,
        rows=pd.DataFrame(rows),
        missing_tickers=tuple(sorted(set(missing_tickers))),
        configured_count=len(config.leadership_etfs),
        latest_dates=latest_dates,
    )
    validate_ranking_inputs(latest)
    if latest.missing_tickers:
        LOGGER.warning(
            "Sector ETF ranking date %s is missing ticker(s): %s",
            latest.ranking_date,
            ", ".join(latest.missing_tickers),
        )
    return latest


def validate_ranking_inputs(latest: LatestSectorETFMetrics) -> None:
    """Validate that ranking rows all represent the exact selected date."""

    if latest.rows.empty:
        raise SectorETFRankingValidationError(
            f"No ETF metrics are available on ranking date "
            f"{latest.ranking_date}"
        )
    required_columns = [
        "date",
        "ticker",
        "sector_id",
        "sector_name",
        "sector_name_cn",
        "adj_close",
    ]
    for horizon_trading_days in SECTOR_ETF_RETURN_HORIZONS:
        required_columns.extend(
            [
                f"reference_date_{horizon_trading_days}td",
                f"reference_adj_close_{horizon_trading_days}td",
                f"adj_close_return_{horizon_trading_days}td",
            ]
        )
    missing_columns = [
        column for column in required_columns if column not in latest.rows
    ]
    if missing_columns:
        raise SectorETFRankingValidationError(
            "Ranking input missing column(s): " + ", ".join(missing_columns)
        )

    dates = latest.rows["date"].astype(str)
    if not dates.eq(latest.ranking_date).all():
        raise SectorETFRankingValidationError(
            "Ranking input mixes multiple metric dates"
        )
    tickers = (
        latest.rows["ticker"].fillna("").astype(str).str.strip().str.upper()
    )
    if tickers.eq("").any() or tickers.duplicated().any():
        raise SectorETFRankingValidationError(
            "Ranking input tickers must be non-empty and unique"
        )

    adj_close = pd.to_numeric(latest.rows["adj_close"], errors="coerce")
    adj_values = adj_close.to_numpy(dtype=float, na_value=np.nan)
    if (
        adj_close.isna().any()
        or not np.isfinite(adj_values).all()
        or (adj_close <= 0).any()
    ):
        raise SectorETFRankingValidationError(
            "Ranking input adj_close values must be finite and positive"
        )

    for horizon_trading_days in SECTOR_ETF_RETURN_HORIZONS:
        return_column = (
            f"adj_close_return_{horizon_trading_days}td"
        )
        reference_date_column = (
            f"reference_date_{horizon_trading_days}td"
        )
        reference_price_column = (
            f"reference_adj_close_{horizon_trading_days}td"
        )
        raw_returns = latest.rows[return_column]
        returns = pd.to_numeric(raw_returns, errors="coerce")
        invalid_returns = returns.isna() & raw_returns.notna()
        if invalid_returns.any():
            raise SectorETFRankingValidationError(
                f"Ranking input has non-numeric {return_column}"
            )
        finite_returns = returns.dropna().to_numpy(dtype=float)
        if not np.isfinite(finite_returns).all():
            raise SectorETFRankingValidationError(
                f"Ranking input has infinite {return_column}"
            )
        available = returns.notna()
        if not available.any():
            continue
        reference_dates = pd.to_datetime(
            latest.rows[reference_date_column],
            errors="coerce",
            format="mixed",
        )
        reference_prices = pd.to_numeric(
            latest.rows[reference_price_column],
            errors="coerce",
        )
        reference_values = reference_prices.loc[available].to_numpy(
            dtype=float,
            na_value=np.nan,
        )
        if (
            reference_dates.loc[available].isna().any()
            or not np.isfinite(reference_values).all()
            or (reference_values <= 0).any()
        ):
            raise SectorETFRankingValidationError(
                f"Ranking input has invalid {horizon_trading_days}-trading-"
                "day reference "
                "data"
            )
        ranking_dates = pd.to_datetime(
            latest.rows.loc[available, "date"],
            format="mixed",
        )
        if (
            reference_dates.loc[available].to_numpy()
            >= ranking_dates.to_numpy()
        ).any():
            raise SectorETFRankingValidationError(
                f"Ranking input {horizon_trading_days}-trading-day reference "
                "date must precede the ranking date"
            )
        expected_returns = (
            adj_close.loc[available].to_numpy(dtype=float)
            / reference_values
            - 1.0
        )
        if not np.allclose(
            returns.loc[available].to_numpy(dtype=float),
            expected_returns,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise SectorETFRankingValidationError(
                f"Ranking input {horizon_trading_days}-trading-day return "
                "formula is invalid"
            )


def build_horizon_ranking(
    latest_rows: pd.DataFrame,
    horizon_trading_days: int,
) -> pd.DataFrame:
    """Build deterministic top-three and bottom-three rows for one horizon."""

    if horizon_trading_days not in SECTOR_ETF_RETURN_HORIZONS:
        raise ValueError(
            "Unsupported sector ETF return horizon: "
            f"{horizon_trading_days}"
        )
    return_column = f"adj_close_return_{horizon_trading_days}td"
    reference_date_column = (
        f"reference_date_{horizon_trading_days}td"
    )
    reference_price_column = (
        f"reference_adj_close_{horizon_trading_days}td"
    )
    working = latest_rows.copy()
    working[return_column] = pd.to_numeric(
        working[return_column],
        errors="coerce",
    )
    valid = working.loc[working[return_column].notna()].copy()
    universe_size = len(valid)
    if universe_size < MIN_RANKING_UNIVERSE_SIZE:
        raise InsufficientRankingUniverseError(
            f"{horizon_trading_days}-trading-day ranking has only "
            f"{universe_size} valid ETF values; at least "
            f"{MIN_RANKING_UNIVERSE_SIZE} are required"
        )

    top = valid.sort_values(
        [return_column, "ticker"],
        ascending=[False, True],
        kind="stable",
    ).head(3)
    bottom = valid.sort_values(
        [return_column, "ticker"],
        ascending=[True, True],
        kind="stable",
    ).head(3)
    if set(top["ticker"]) & set(bottom["ticker"]):
        raise InsufficientRankingUniverseError(
            f"{horizon_trading_days}-trading-day top and bottom rankings "
            "overlap"
        )

    output_rows: list[dict[str, object]] = []
    for ranking_group, selected in (("top", top), ("bottom", bottom)):
        for rank, row in enumerate(selected.itertuples(index=False), start=1):
            output_rows.append(
                {
                    "date": str(row.date),
                    "horizon_trading_days": horizon_trading_days,
                    "ranking_group": ranking_group,
                    "rank": rank,
                    "ticker": str(row.ticker),
                    "sector_id": str(row.sector_id),
                    "sector_name": str(row.sector_name),
                    "sector_name_cn": str(row.sector_name_cn),
                    "adj_close": float(row.adj_close),
                    "reference_date": str(
                        getattr(row, reference_date_column)
                    ),
                    "reference_adj_close": float(
                        getattr(row, reference_price_column)
                    ),
                    "adj_close_return": float(
                        getattr(row, return_column)
                    ),
                    "universe_size": universe_size,
                }
            )
    return pd.DataFrame(output_rows, columns=RANKING_COLUMNS)


def _sort_rankings(rankings: pd.DataFrame) -> pd.DataFrame:
    output = rankings.copy()
    for column in (
        "horizon_trading_days",
        "rank",
        "universe_size",
    ):
        output[column] = pd.to_numeric(
            output[column],
            errors="raise",
        ).astype(int)
    for column in (
        "adj_close",
        "reference_adj_close",
        "adj_close_return",
    ):
        output[column] = pd.to_numeric(
            output[column],
            errors="raise",
        ).astype(float)
    for column in (
        "date",
        "ranking_group",
        "ticker",
        "sector_id",
        "sector_name",
        "sector_name_cn",
        "reference_date",
    ):
        output[column] = output[column].astype(str)
    horizon_order = {
        horizon: index
        for index, horizon in enumerate(SECTOR_ETF_RETURN_HORIZONS)
    }
    group_order = {
        group: index for index, group in enumerate(RANKING_GROUP_ORDER)
    }
    output["_horizon_order"] = output["horizon_trading_days"].map(
        horizon_order
    )
    output["_group_order"] = output["ranking_group"].map(group_order)
    output = output.sort_values(
        ["date", "_horizon_order", "_group_order", "rank"],
        kind="stable",
    )
    return output.drop(
        columns=["_horizon_order", "_group_order"]
    ).reset_index(drop=True)


def build_daily_sector_etf_rankings(
    latest: LatestSectorETFMetrics,
) -> pd.DataFrame:
    """Build all three independent horizon rankings for one exact date."""

    validate_ranking_inputs(latest)
    rankings = pd.concat(
        [
            build_horizon_ranking(latest.rows, horizon_trading_days)
            for horizon_trading_days in SECTOR_ETF_RETURN_HORIZONS
        ],
        ignore_index=True,
    )
    rankings = _sort_rankings(rankings).reindex(columns=RANKING_COLUMNS)
    validate_sector_etf_rankings(rankings, require_single_date=True)
    return rankings


def validate_sector_etf_rankings(
    rankings: pd.DataFrame,
    *,
    require_single_date: bool = False,
) -> None:
    """Validate ranking schema, keys, ordering fields, and return formulas."""

    if list(rankings.columns) != RANKING_COLUMNS:
        raise SectorETFRankingValidationError(
            f"Ranking schema mismatch: expected {RANKING_COLUMNS}, found "
            f"{list(rankings.columns)}"
        )
    if rankings.empty:
        raise SectorETFRankingValidationError("Ranking data cannot be empty")

    parsed_dates = pd.to_datetime(
        rankings["date"],
        errors="coerce",
        format="mixed",
    )
    if parsed_dates.isna().any():
        raise SectorETFRankingValidationError(
            "Ranking data contains invalid dates"
        )
    if require_single_date and rankings["date"].astype(str).nunique() != 1:
        raise SectorETFRankingValidationError(
            "Daily ranking output must contain exactly one date"
        )

    horizons = pd.to_numeric(
        rankings["horizon_trading_days"],
        errors="coerce",
    )
    ranks = pd.to_numeric(rankings["rank"], errors="coerce")
    universe_sizes = pd.to_numeric(
        rankings["universe_size"],
        errors="coerce",
    )
    if horizons.isna().any() or not set(horizons.astype(int)).issubset(
        set(SECTOR_ETF_RETURN_HORIZONS)
    ):
        raise SectorETFRankingValidationError(
            "Ranking data contains unsupported horizon_trading_days"
        )
    if ranks.isna().any() or not set(ranks.astype(int)).issubset({1, 2, 3}):
        raise SectorETFRankingValidationError(
            "Ranking data rank must be 1, 2, or 3"
        )
    if (
        universe_sizes.isna().any()
        or (universe_sizes < MIN_RANKING_UNIVERSE_SIZE).any()
    ):
        raise SectorETFRankingValidationError(
            "Ranking data universe_size is invalid"
        )
    if not set(rankings["ranking_group"]).issubset(
        set(RANKING_GROUP_ORDER)
    ):
        raise SectorETFRankingValidationError(
            "Ranking data contains invalid ranking_group"
        )

    key_columns = [
        "date",
        "horizon_trading_days",
        "ranking_group",
        "rank",
    ]
    if rankings.duplicated(key_columns).any():
        raise SectorETFRankingValidationError(
            "Ranking data contains duplicate primary keys"
        )

    numeric_columns = [
        "adj_close",
        "reference_adj_close",
        "adj_close_return",
    ]
    numeric = rankings[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if numeric.isna().any().any() or not np.isfinite(
        numeric.to_numpy(dtype=float)
    ).all():
        raise SectorETFRankingValidationError(
            "Ranking prices and returns must be finite numeric values"
        )
    if (
        numeric["adj_close"].le(0).any()
        or numeric["reference_adj_close"].le(0).any()
    ):
        raise SectorETFRankingValidationError(
            "Ranking adjusted close prices must be positive"
        )
    expected_returns = (
        numeric["adj_close"] / numeric["reference_adj_close"] - 1.0
    )
    if not np.allclose(
        numeric["adj_close_return"],
        expected_returns,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise SectorETFRankingValidationError(
            "Ranking adjusted-close return formula is invalid"
        )

    reference_dates = pd.to_datetime(
        rankings["reference_date"],
        errors="coerce",
        format="mixed",
    )
    if reference_dates.isna().any():
        raise SectorETFRankingValidationError(
            "Ranking data contains invalid reference dates"
        )
    if (reference_dates >= parsed_dates).any():
        raise SectorETFRankingValidationError(
            "Ranking reference date must precede its ranking date"
        )

    daily_keys = ["date", "horizon_trading_days"]
    universe_counts = rankings.groupby(
        daily_keys,
        sort=False,
    )["universe_size"].nunique()
    if universe_counts.ne(1).any():
        raise SectorETFRankingValidationError(
            "Each date/horizon must have one universe_size"
        )

    group_keys = [*daily_keys, "ranking_group"]
    grouped = rankings.groupby(group_keys, sort=False)
    group_sizes = grouped.size()
    rank_counts = grouped["rank"].nunique()
    rank_minimums = grouped["rank"].min()
    rank_maximums = grouped["rank"].max()
    if (
        group_sizes.ne(3).any()
        or rank_counts.ne(3).any()
        or rank_minimums.ne(1).any()
        or rank_maximums.ne(3).any()
    ):
        raise SectorETFRankingValidationError(
            "Each date/horizon/group must contain ranks 1, 2, and 3"
        )

    for ranking_group in RANKING_GROUP_ORDER:
        selected = rankings.loc[
            rankings["ranking_group"].eq(ranking_group)
        ]
        ordered = selected.sort_values(
            [*daily_keys, "rank"],
            kind="stable",
        )
        expected = selected.sort_values(
            [*daily_keys, "adj_close_return", "ticker"],
            ascending=[
                True,
                True,
                ranking_group == "bottom",
                True,
            ],
            kind="stable",
        )
        if list(ordered["ticker"]) != list(expected["ticker"]):
            raise SectorETFRankingValidationError(
                f"{ranking_group} rank order is inconsistent with return "
                "and ticker tie-break"
            )

    top = rankings.loc[
        rankings["ranking_group"].eq("top"),
        [*daily_keys, "ticker"],
    ]
    bottom = rankings.loc[
        rankings["ranking_group"].eq("bottom"),
        [*daily_keys, "ticker"],
    ]
    if not top.merge(
        bottom,
        on=[*daily_keys, "ticker"],
        how="inner",
    ).empty:
        raise SectorETFRankingValidationError(
            "Top and bottom ranking groups cannot overlap"
        )


def _canonical_csv_text(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    float_format: str | None = None,
) -> str:
    buffer = StringIO()
    frame.to_csv(
        buffer,
        columns=columns,
        index=False,
        encoding="utf-8",
        float_format=float_format,
        na_rep="",
        lineterminator="\n",
    )
    return buffer.getvalue()


def _write_csv_if_changed(
    frame: pd.DataFrame,
    path: Path,
    columns: list[str],
    *,
    float_format: str | None = None,
) -> bool:
    expected_content = _canonical_csv_text(
        frame,
        columns,
        float_format=float_format,
    )
    if path.exists() and path.read_text(encoding="utf-8") == expected_content:
        return False

    def verify_temp_file(temp_path: Path) -> None:
        if temp_path.read_text(encoding="utf-8") != expected_content:
            raise SectorETFRankingValidationError(
                f"Temporary CSV verification failed: {temp_path}"
            )

    atomic_write_csv(
        frame,
        path,
        columns,
        float_format=float_format,
        na_rep="",
        lineterminator="\n",
        validate_temp_file=verify_temp_file,
    )
    return True


def load_sector_etf_ranking_history(
    history_file: Path | str = DEFAULT_RANKING_HISTORY_FILE,
) -> pd.DataFrame:
    """Load and validate existing long-format daily ranking history."""

    path = Path(history_file)
    if not path.exists():
        return pd.DataFrame(columns=RANKING_COLUMNS)
    try:
        history = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=RANKING_COLUMNS)
    if list(history.columns) != RANKING_COLUMNS:
        raise SectorETFRankingValidationError(
            f"Ranking history schema mismatch for {path}"
        )
    if not history.empty:
        validate_sector_etf_rankings(history)
    return _sort_rankings(history).reindex(columns=RANKING_COLUMNS)


def upsert_sector_etf_ranking_history(
    rankings: pd.DataFrame,
    history_file: Path | str = DEFAULT_RANKING_HISTORY_FILE,
) -> bool:
    """Replace one ranking date atomically or append a new date."""

    validate_sector_etf_rankings(rankings, require_single_date=True)
    ranking_date = str(rankings["date"].iloc[0])
    path = Path(history_file)
    existing = load_sector_etf_ranking_history(path)
    retained = existing.loc[
        ~existing["date"].astype(str).eq(ranking_date)
    ]
    combined = _sort_rankings(
        pd.concat([retained, rankings], ignore_index=True)
    ).reindex(columns=RANKING_COLUMNS)
    validate_sector_etf_rankings(combined)
    return _write_csv_if_changed(
        combined,
        path,
        RANKING_COLUMNS,
        float_format=METRICS_FLOAT_FORMAT,
    )


def rebuild_sector_etf_ranking_history(
    *,
    config_path: Path | str = DEFAULT_CONFIG_FILE,
    metrics_dir: Path | str = DEFAULT_METRICS_DIR,
    ranking_history_file: Path | str = DEFAULT_RANKING_HISTORY_FILE,
) -> SectorETFRankingHistoryRebuildSummary:
    """
    Replace ranking history from local metrics without sending email.

    Only dates present in every configured leadership ETF metrics file are
    considered. A common date is skipped when any horizon has fewer than six
    valid returns, because disjoint Top 3 and Bottom 3 groups would be
    impossible.
    """

    config = load_sector_etf_config(config_path)
    loaded: dict[str, pd.DataFrame] = {}
    for etf in config.leadership_etfs:
        path = resolve_sector_etf_metrics_path(metrics_dir, etf)
        if not path.exists():
            raise FileNotFoundError(
                f"Metrics file not found for {etf.ticker}: {path}"
            )
        loaded[etf.ticker] = _load_one_metrics_file(path, etf.ticker)

    common_dates = sorted(
        set.intersection(
            *(
                set(frame["date"].astype(str))
                for frame in loaded.values()
            )
        )
    )
    if not common_dates:
        raise SectorETFRankingValidationError(
            "Leadership metrics have no common trading date"
        )

    common_date_set = set(common_dates)
    enriched_frames: list[pd.DataFrame] = []
    for etf in config.leadership_etfs:
        frame = loaded[etf.ticker]
        selected = frame.loc[
            frame["date"].astype(str).isin(common_date_set)
        ].copy()
        selected["date"] = selected["date"].astype(str)
        selected["ticker"] = etf.ticker
        selected["sector_id"] = etf.sector_id
        selected["sector_name"] = etf.sector_name
        selected["sector_name_cn"] = etf.sector_name_cn
        enriched_frames.append(selected)
    combined = pd.concat(enriched_frames, ignore_index=True)

    horizon_frames: dict[int, pd.DataFrame] = {}
    rankable_dates = set(common_dates)
    for horizon_trading_days in SECTOR_ETF_RETURN_HORIZONS:
        suffix = f"{horizon_trading_days}td"
        working = combined[
            [
                "date",
                "ticker",
                "sector_id",
                "sector_name",
                "sector_name_cn",
                "adj_close",
                f"reference_date_{suffix}",
                f"reference_adj_close_{suffix}",
                f"adj_close_return_{suffix}",
            ]
        ].rename(
            columns={
                f"reference_date_{suffix}": "reference_date",
                f"reference_adj_close_{suffix}": "reference_adj_close",
                f"adj_close_return_{suffix}": "adj_close_return",
            }
        )
        working["adj_close_return"] = pd.to_numeric(
            working["adj_close_return"],
            errors="coerce",
        )
        valid = working.loc[
            working["adj_close_return"].notna()
        ].copy()
        universe_sizes = valid.groupby("date", sort=False).size()
        eligible_dates = set(
            universe_sizes.loc[
                universe_sizes >= MIN_RANKING_UNIVERSE_SIZE
            ].index.astype(str)
        )
        rankable_dates &= eligible_dates
        valid["universe_size"] = (
            valid["date"].map(universe_sizes).astype(int)
        )
        horizon_frames[horizon_trading_days] = valid

    if not rankable_dates:
        raise InsufficientRankingUniverseError(
            "No common trading date has enough valid ETFs for every horizon"
        )

    ranked_groups: list[pd.DataFrame] = []
    for horizon_trading_days in SECTOR_ETF_RETURN_HORIZONS:
        valid = horizon_frames[horizon_trading_days].loc[
            horizon_frames[horizon_trading_days]["date"].isin(
                rankable_dates
            )
        ]
        for ranking_group in RANKING_GROUP_ORDER:
            selected = valid.sort_values(
                ["date", "adj_close_return", "ticker"],
                ascending=[
                    True,
                    ranking_group == "bottom",
                    True,
                ],
                kind="stable",
            )
            selected = (
                selected.groupby("date", sort=False)
                .head(3)
                .copy()
            )
            selected["ranking_group"] = ranking_group
            selected["rank"] = (
                selected.groupby("date", sort=False).cumcount() + 1
            )
            selected["horizon_trading_days"] = horizon_trading_days
            ranked_groups.append(selected.reindex(columns=RANKING_COLUMNS))

    history = _sort_rankings(
        pd.concat(ranked_groups, ignore_index=True)
    ).reindex(columns=RANKING_COLUMNS)
    validate_sector_etf_rankings(history)
    history_path = Path(ranking_history_file)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_written = _write_csv_if_changed(
        history,
        history_path,
        RANKING_COLUMNS,
        float_format=METRICS_FLOAT_FORMAT,
    )
    return SectorETFRankingHistoryRebuildSummary(
        configured_etfs=len(config.leadership_etfs),
        common_dates=len(common_dates),
        ranked_dates=int(history["date"].nunique()),
        skipped_unrankable_dates=len(common_dates) - len(rankable_dates),
        ranking_rows=len(history),
        earliest_date=str(history["date"].iloc[0]),
        latest_date=str(history["date"].iloc[-1]),
        history_written=history_written,
    )


def _format_return(value: object) -> str:
    return f"{float(value):.2%}"


def _plain_ranking_table(
    rankings: pd.DataFrame,
    horizon_trading_days: int,
    ranking_group: str,
) -> str:
    selected = rankings.loc[
        rankings["horizon_trading_days"].eq(horizon_trading_days)
        & rankings["ranking_group"].eq(ranking_group)
    ].sort_values("rank")
    lines = [
        "Rank | Ticker | Sector / Industry | Return | Adjusted Close | "
        "Reference Date | Reference Adjusted Close"
    ]
    for row in selected.itertuples():
        lines.append(
            f"{row.rank} | {row.ticker} | {row.sector_name} | "
            f"{_format_return(row.adj_close_return)} | "
            f"{row.adj_close:.4f} | {row.reference_date} | "
            f"{row.reference_adj_close:.4f}"
        )
    return "\n".join(lines)


def _html_ranking_table(
    rankings: pd.DataFrame,
    horizon_trading_days: int,
    ranking_group: str,
) -> str:
    selected = rankings.loc[
        rankings["horizon_trading_days"].eq(horizon_trading_days)
        & rankings["ranking_group"].eq(ranking_group)
    ].sort_values("rank")
    rows = []
    for row in selected.itertuples():
        cells = [
            row.rank,
            row.ticker,
            row.sector_name,
            _format_return(row.adj_close_return),
            f"{row.adj_close:.4f}",
            row.reference_date,
            f"{row.reference_adj_close:.4f}",
        ]
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(cell))}</td>" for cell in cells
            )
            + "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Rank</th><th>Ticker</th><th>Sector / Industry</th><th>Return</th>"
        "<th>Adjusted Close</th><th>Reference Date</th>"
        "<th>Reference Adjusted Close</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def format_sector_etf_ranking_email(
    rankings: pd.DataFrame,
    *,
    participating_count: int,
    configured_count: int,
    missing_tickers: Sequence[str] = (),
) -> SectorETFRankingEmail:
    """Render deterministic plain-text and HTML ranking email alternatives."""

    validate_sector_etf_rankings(rankings, require_single_date=True)
    ranking_date = str(rankings["date"].iloc[0])
    horizon_sizes = {
        int(horizon): int(group["universe_size"].iloc[0])
        for horizon, group in rankings.groupby("horizon_trading_days")
    }
    incomplete = (
        participating_count != configured_count
        or bool(missing_tickers)
        or any(
            horizon_sizes.get(horizon, 0) != configured_count
            for horizon in SECTOR_ETF_RETURN_HORIZONS
        )
    )
    subject_prefix = (
        "[Investment OS][INCOMPLETE]"
        if incomplete
        else "[Investment OS]"
    )
    subject = (
        f"{subject_prefix} Sector ETF Rotation Rankings - {ranking_date}"
    )
    completeness = "INCOMPLETE" if incomplete else "Complete"
    missing_text = (
        ", ".join(sorted(set(missing_tickers))) if missing_tickers else "N/A"
    )
    header_lines = [
        f"Ranking Date: {ranking_date}",
        f"Leadership Universe Size: {configured_count}",
        (
            "Leadership Universe: 11 primary sectors + SOXX semiconductors "
            "+ IGV software"
        ),
        f"Participating ETFs: {participating_count}/{configured_count}",
        f"Data Completeness: {completeness}",
        f"Missing Tickers: {missing_text}",
        "Price Source: Yahoo Finance Adjusted Close",
        (
            "Return Definition: Current Adj Close / Reference Adj Close - 1"
        ),
        (
            "Reference Rule: Fixed trading-session lookback within each "
            "ETF's own observed Yahoo price history"
        ),
    ]
    plain_sections = ["\n".join(header_lines)]
    html_sections = [
        "<h1>Sector and Industry ETF Leadership Rankings</h1>",
        "<ul>"
        + "".join(
            f"<li>{html.escape(line)}</li>" for line in header_lines
        )
        + "</ul>",
    ]
    for horizon_trading_days in EMAIL_HORIZON_ORDER:
        plain_sections.append(
            f"{horizon_trading_days}-Trading-Day Return\n"
            f"Valid ETFs: {horizon_sizes[horizon_trading_days]}"
        )
        html_sections.append(
            f"<h2>{horizon_trading_days}-Trading-Day Return</h2>"
            f"<p>Valid ETFs: {horizon_sizes[horizon_trading_days]}</p>"
        )
        for ranking_group, label in (("top", "Top 3"), ("bottom", "Bottom 3")):
            plain_sections.append(
                f"{label}\n"
                + _plain_ranking_table(
                    rankings,
                    horizon_trading_days,
                    ranking_group,
                )
            )
            html_sections.append(f"<h3>{label}</h3>")
            html_sections.append(
                _html_ranking_table(
                    rankings,
                    horizon_trading_days,
                    ranking_group,
                )
            )

    footer = (
        "Returns are calculated from adjusted close prices over fixed "
        "trading-session lookbacks. A 30-trading-day return uses the "
        "adjusted close from 30 ETF trading observations earlier.\n"
        "This is a quantitative ranking summary, not an investment "
        "recommendation."
    )
    plain_sections.append(footer)
    html_sections.append(
        "<p>Returns are calculated from adjusted close prices over fixed "
        "trading-session lookbacks. A 30-trading-day return uses the "
        "adjusted close from 30 ETF trading observations earlier.<br>"
        "This is a quantitative ranking summary, not an investment "
        "recommendation.</p>"
    )
    html_body = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<style>"
        "body{font-family:Arial,sans-serif;color:#222}"
        "table{border-collapse:collapse;margin-bottom:18px}"
        "th,td{border:1px solid #ccc;padding:6px 8px;text-align:left}"
        "th{background:#f3f4f6}"
        "</style></head><body>"
        + "".join(html_sections)
        + "</body></html>"
    )
    return SectorETFRankingEmail(
        subject=subject,
        plain_text="\n\n".join(plain_sections),
        html=html_body,
        incomplete=incomplete,
    )


def format_sector_etf_test_email(
    production_email: SectorETFRankingEmail,
    *,
    ranking_date: str,
) -> SectorETFRankingEmail:
    """Add test-only subject and banner without duplicating the renderer."""

    subject = (
        f"[TEST][INCOMPLETE][Investment OS] "
        f"Sector ETF Leadership Rankings - {ranking_date}"
        if production_email.incomplete
        else (
            f"[TEST][Investment OS] Sector ETF Leadership Rankings - "
            f"{ranking_date}"
        )
    )
    body_tag = "<body>"
    if body_tag not in production_email.html:
        raise SectorETFRankingValidationError(
            "Rendered HTML email has no body element"
        )
    return SectorETFRankingEmail(
        subject=subject,
        plain_text=(
            TEST_EMAIL_BANNER_PLAIN
            + "\n\n"
            + production_email.plain_text
        ),
        html=production_email.html.replace(
            body_tag,
            body_tag + TEST_EMAIL_BANNER_HTML,
            1,
        ),
        incomplete=production_email.incomplete,
    )


def validate_sector_etf_test_email_preview(
    email_message: SectorETFRankingEmail,
    rankings: pd.DataFrame,
    *,
    ranking_date: str,
) -> None:
    """Fail closed before a real test send if the preview is malformed."""

    validate_sector_etf_rankings(rankings, require_single_date=True)
    if not email_message.subject.startswith("[TEST]"):
        raise SectorETFRankingValidationError(
            "Test email subject must start with [TEST]"
        )
    if ranking_date not in email_message.subject:
        raise SectorETFRankingValidationError(
            "Test email subject does not contain the ranking date"
        )
    for content_name, content in (
        ("plain text", email_message.plain_text),
        ("HTML", email_message.html),
    ):
        if TEST_EMAIL_BANNER_PLAIN.splitlines()[0] not in content:
            raise SectorETFRankingValidationError(
                f"Test banner is missing from {content_name}"
            )
        positions = [
            content.find(f"{horizon}-Trading-Day Return")
            for horizon in EMAIL_HORIZON_ORDER
        ]
        if any(position < 0 for position in positions) or positions != sorted(
            positions
        ):
            raise SectorETFRankingValidationError(
                f"Test email {content_name} horizon order is invalid"
            )
        if content.count("Top 3") != 3 or content.count("Bottom 3") != 3:
            raise SectorETFRankingValidationError(
                f"Test email {content_name} ranking groups are incomplete"
            )
        lowered = content.casefold()
        if any(marker in lowered for marker in ("none%", "nan%")):
            raise SectorETFRankingValidationError(
                f"Test email {content_name} contains an invalid return"
            )
        if f"Ranking Date: {ranking_date}" not in content:
            raise SectorETFRankingValidationError(
                f"Test email {content_name} ranking date is inconsistent"
            )

        for index, horizon_trading_days in enumerate(EMAIL_HORIZON_ORDER):
            start = positions[index]
            end = (
                positions[index + 1]
                if index + 1 < len(positions)
                else len(content)
            )
            section = content[start:end]
            if section.find("Top 3") > section.find("Bottom 3"):
                raise SectorETFRankingValidationError(
                    f"Test email {content_name} "
                    f"{horizon_trading_days}td group order is invalid"
                )

    if email_message.html.count("<table>") != 6 or (
        email_message.html.count("</table>") != 6
    ):
        raise SectorETFRankingValidationError(
            "Test email HTML tables are unbalanced"
        )
    for row in rankings.itertuples(index=False):
        percentage = _format_return(row.adj_close_return)
        for content_name, content in (
            ("plain text", email_message.plain_text),
            ("HTML", email_message.html),
        ):
            if str(row.ticker) not in content or percentage not in content:
                raise SectorETFRankingValidationError(
                    f"Test email {content_name} is missing ranking values"
                )


def load_sector_etf_rankings_for_test_email(
    ranking_history_file: Path | str = DEFAULT_RANKING_HISTORY_FILE,
    *,
    ranking_date: str | None = None,
) -> pd.DataFrame:
    """Read one already-persisted ranking date without rewriting history."""

    history = load_sector_etf_ranking_history(ranking_history_file)
    if history.empty:
        raise SectorETFRankingValidationError(
            "No persisted sector ETF rankings are available for test email"
        )
    selected_date = (
        _parse_ranking_date(ranking_date, option_name="ranking_date")
        if ranking_date is not None
        else str(history["date"].max())
    )
    selected = history.loc[
        history["date"].astype(str).eq(selected_date)
    ].copy()
    if selected.empty:
        raise SectorETFRankingValidationError(
            f"Ranking date is not present in ranking history: {selected_date}"
        )
    selected = _sort_rankings(selected).reindex(columns=RANKING_COLUMNS)
    validate_sector_etf_rankings(selected, require_single_date=True)
    return selected


def run_sector_etf_test_email(
    *,
    config_path: Path | str = DEFAULT_CONFIG_FILE,
    metrics_dir: Path | str = DEFAULT_METRICS_DIR,
    ranking_history_file: Path | str = DEFAULT_RANKING_HISTORY_FILE,
    ranking_date: str | None = None,
    email_sender: Callable[..., int] = send_email,
) -> SectorETFTestEmailResult:
    """Send one marked test email without touching production idempotency."""

    config = load_sector_etf_config(config_path)
    rankings = load_sector_etf_rankings_for_test_email(
        ranking_history_file,
        ranking_date=ranking_date,
    )
    selected_date = str(rankings["date"].iloc[0])
    latest = load_latest_sector_etf_metrics(
        config,
        metrics_dir=metrics_dir,
        ranking_date=selected_date,
    )
    production_email = format_sector_etf_ranking_email(
        rankings,
        participating_count=latest.participating_count,
        configured_count=latest.configured_count,
        missing_tickers=latest.missing_tickers,
    )
    test_email = format_sector_etf_test_email(
        production_email,
        ranking_date=selected_date,
    )
    validate_sector_etf_test_email_preview(
        test_email,
        rankings,
        ranking_date=selected_date,
    )
    recipient_count = int(
        email_sender(
            subject=test_email.subject,
            body=test_email.plain_text,
            html_body=test_email.html,
        )
        or 0
    )
    if recipient_count <= 0:
        raise RuntimeError(
            "Test email sender reported no configured recipients"
        )
    return SectorETFTestEmailResult(
        ranking_date=selected_date,
        configured_etfs=latest.configured_count,
        participating_etfs=latest.participating_count,
        missing_tickers=latest.missing_tickers,
        recipient_count=recipient_count,
        email=test_email,
    )


def load_sector_etf_email_log(
    email_log_file: Path | str = DEFAULT_EMAIL_LOG_FILE,
) -> pd.DataFrame:
    """Load and validate the one-row-per-ranking-date email status log."""

    path = Path(email_log_file)
    if not path.exists():
        return pd.DataFrame(columns=EMAIL_LOG_COLUMNS)
    try:
        log = pd.read_csv(path, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=EMAIL_LOG_COLUMNS)
    if list(log.columns) != EMAIL_LOG_COLUMNS:
        raise SectorETFRankingValidationError(
            f"Email log schema mismatch for {path}"
        )
    if log["ranking_date"].duplicated().any():
        raise SectorETFRankingValidationError(
            "Email log contains duplicate ranking_date rows"
        )
    parsed_dates = pd.to_datetime(
        log["ranking_date"],
        errors="coerce",
        format="mixed",
    )
    if parsed_dates.isna().any():
        raise SectorETFRankingValidationError(
            "Email log contains invalid ranking_date"
        )
    if not set(log["status"]).issubset({"success", "error"}):
        raise SectorETFRankingValidationError(
            "Email log status must be success or error"
        )
    recipient_counts = pd.to_numeric(
        log["recipient_count"],
        errors="coerce",
    )
    if recipient_counts.isna().any() or (recipient_counts < 0).any():
        raise SectorETFRankingValidationError(
            "Email log recipient_count must be non-negative"
        )
    log["recipient_count"] = recipient_counts.astype(int)
    return log.sort_values("ranking_date", kind="stable").reset_index(drop=True)


def _upsert_sector_etf_email_log(
    *,
    ranking_date: str,
    status: str,
    recipient_count: int,
    error_message: str,
    sent_at_utc: str,
    email_log_file: Path | str,
) -> None:
    log_path = Path(email_log_file)
    existing = load_sector_etf_email_log(log_path)
    retained = existing.loc[
        ~existing["ranking_date"].astype(str).eq(ranking_date)
    ]
    new_row = pd.DataFrame(
        [
            {
                "ranking_date": ranking_date,
                "sent_at_utc": sent_at_utc,
                "status": status,
                "recipient_count": recipient_count,
                "error_message": error_message,
            }
        ]
    )
    combined = pd.concat([retained, new_row], ignore_index=True)
    combined = combined.sort_values(
        "ranking_date",
        kind="stable",
    ).reset_index(drop=True)
    _write_csv_if_changed(combined, log_path, EMAIL_LOG_COLUMNS)


def _now_utc_iso() -> str:
    return (
        datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def send_sector_etf_ranking_email(
    email_message: SectorETFRankingEmail,
    *,
    ranking_date: str,
    email_log_file: Path | str = DEFAULT_EMAIL_LOG_FILE,
    force_email: bool = False,
    email_sender: Callable[..., int] = send_email,
    now_utc: Callable[[], str] = _now_utc_iso,
) -> RankingEmailSendResult:
    """Send once per successful ranking date, while allowing error retries."""

    log = load_sector_etf_email_log(email_log_file)
    previous_success = (
        log["ranking_date"].astype(str).eq(ranking_date)
        & log["status"].eq("success")
    ).any()
    if previous_success and not force_email:
        LOGGER.info(
            "Skipping duplicate sector ETF ranking email for %s",
            ranking_date,
        )
        return RankingEmailSendResult(status="skipped_duplicate")

    sent_at_utc = now_utc()
    try:
        recipient_count = int(
            email_sender(
                subject=email_message.subject,
                body=email_message.plain_text,
                html_body=email_message.html,
            )
            or 0
        )
    except Exception as error:
        message = short_error(error)
        _upsert_sector_etf_email_log(
            ranking_date=ranking_date,
            status="error",
            recipient_count=0,
            error_message=message,
            sent_at_utc=sent_at_utc,
            email_log_file=email_log_file,
        )
        LOGGER.error(
            "Sector ETF ranking email failed for %s: %s",
            ranking_date,
            message,
        )
        return RankingEmailSendResult(
            status="error",
            error_message=message,
        )

    _upsert_sector_etf_email_log(
        ranking_date=ranking_date,
        status="success",
        recipient_count=recipient_count,
        error_message="",
        sent_at_utc=sent_at_utc,
        email_log_file=email_log_file,
    )
    return RankingEmailSendResult(
        status="success",
        recipient_count=recipient_count,
    )


def run_sector_etf_daily_ranking(
    *,
    config_path: Path | str = DEFAULT_CONFIG_FILE,
    metrics_dir: Path | str = DEFAULT_METRICS_DIR,
    ranking_history_file: Path | str = DEFAULT_RANKING_HISTORY_FILE,
    email_log_file: Path | str = DEFAULT_EMAIL_LOG_FILE,
    ranking_date: str | None = None,
    send_email_message: bool = False,
    force_email: bool = False,
    email_sender: Callable[..., int] = send_email,
) -> SectorETFRankingSummary:
    """Build and save one date, then optionally send its idempotent email."""

    if force_email and not send_email_message:
        raise ValueError("force_email requires send_email_message=True")
    config = load_sector_etf_config(config_path)
    latest = load_latest_sector_etf_metrics(
        config,
        metrics_dir=metrics_dir,
        ranking_date=ranking_date,
    )
    rankings = build_daily_sector_etf_rankings(latest)
    history_written = upsert_sector_etf_ranking_history(
        rankings,
        ranking_history_file,
    )
    email_message = format_sector_etf_ranking_email(
        rankings,
        participating_count=latest.participating_count,
        configured_count=latest.configured_count,
        missing_tickers=latest.missing_tickers,
    )
    email_result = RankingEmailSendResult(status="not_requested")
    if send_email_message:
        email_result = send_sector_etf_ranking_email(
            email_message,
            ranking_date=latest.ranking_date,
            email_log_file=email_log_file,
            force_email=force_email,
            email_sender=email_sender,
        )
    return SectorETFRankingSummary(
        ranking_date=latest.ranking_date,
        configured_etfs=latest.configured_count,
        participating_etfs=latest.participating_count,
        missing_tickers=latest.missing_tickers,
        ranking_rows=len(rankings),
        history_written=history_written,
        email_status=email_result.status,
        email_error=email_result.error_message,
        rankings=rankings,
        email=email_message,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build local sector ETF 30/90/250-trading-day rankings and "
            "optionally email them."
        )
    )
    email_mode = parser.add_mutually_exclusive_group()
    email_mode.add_argument(
        "--send-email",
        action="store_true",
        help="Send the ranking email after saving local history.",
    )
    email_mode.add_argument(
        "--dry-run-email",
        action="store_true",
        help="Render and print both email alternatives without sending.",
    )
    email_mode.add_argument(
        "--test-email",
        action="store_true",
        help=(
            "Send one marked format-validation email without updating "
            "production ranking history or email logs."
        ),
    )
    email_mode.add_argument(
        "--rebuild-history",
        action="store_true",
        help=(
            "Replace complete local ranking history from local metrics "
            "without sending email or updating the production email log."
        ),
    )
    parser.add_argument(
        "--force-email",
        action="store_true",
        help="Allow resending a ranking date that already succeeded.",
    )
    parser.add_argument(
        "--ranking-date",
        help="Exact metrics trading date in YYYY-MM-DD format.",
    )
    args = parser.parse_args(argv)
    if args.test_email and args.force_email:
        parser.error("--test-email cannot be combined with --force-email")
    if args.force_email and not args.send_email:
        parser.error("--force-email requires --send-email")
    if args.rebuild_history and args.ranking_date:
        parser.error("--rebuild-history cannot be combined with --ranking-date")
    if args.ranking_date:
        try:
            _parse_ranking_date(
                args.ranking_date,
                option_name="--ranking-date",
            )
        except SectorETFRankingValidationError as error:
            parser.error(str(error))
    return args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    try:
        if args.rebuild_history:
            rebuild_summary = rebuild_sector_etf_ranking_history()
            print(rebuild_summary.format())
            return
        if args.test_email:
            test_result = run_sector_etf_test_email(
                ranking_date=args.ranking_date,
            )
            print(test_result.format())
            return
        summary = run_sector_etf_daily_ranking(
            ranking_date=args.ranking_date,
            send_email_message=args.send_email,
            force_email=args.force_email,
        )
    except Exception as error:
        print(
            f"Sector ETF ranking failed: {short_error(error)}",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    print(summary.format())
    if args.dry_run_email:
        print("\nPlain-text email preview:\n")
        print(summary.email.plain_text)
        print("\nHTML email preview:\n")
        print(summary.email.html)
    if summary.email_status == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

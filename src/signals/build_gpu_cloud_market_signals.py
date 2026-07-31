"""Build auditable single-provider GPU cloud market signals."""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd

from market_data.gpu_cloud_config import (
    AVAILABILITY_SCOPE,
    DATA_QUALITY_STATUSES,
    GPU_CLOUD_FETCH_LOG_COLUMNS,
    GPU_CLOUD_FETCH_LOG_FILE,
    GPU_CLOUD_HISTORY_COLUMNS,
    GPU_CLOUD_HISTORY_FILE,
    GPU_CLOUD_SIGNAL_COLUMNS,
    GPU_CLOUD_SIGNALS_FILE,
    INVENTORY_SCOPE,
    SUPPLY_SIGNAL_STATUSES,
    TRACKED_GPU_MODELS,
    TREND_REFERENCE_TOLERANCE_DAYS,
    VAST_MIN_RELIABILITY,
    VAST_PROVIDER,
)
from market_data.gpu_cloud_status import (
    API_KEY_MISSING,
    DAILY_SELECTION_METHOD,
    LEGACY_PARTIAL,
    NO_MARKET_DATA,
    PROVIDER_ERROR,
    SCHEMA_ERROR,
    is_snapshot_eligible_for_signals,
    normalize_snapshot_warnings,
)
from utils.csv_utils import atomic_write_csv


LOGGER = logging.getLogger(__name__)
CSV_FLOAT_FORMAT = "%.15g"


@dataclass(frozen=True)
class GPUCloudSignalsSummary:
    output_file: Path
    rows: int
    dates: int
    latest_date: str
    file_written: bool
    status_counts: dict[str, int]

    def format(self) -> str:
        statuses = ", ".join(
            f"{status}={count}"
            for status, count in sorted(self.status_counts.items())
        )
        return "\n".join(
            [
                "GPU cloud market signals summary:",
                f"- rows: {self.rows}",
                f"- dates: {self.dates}",
                f"- latest date: {self.latest_date}",
                f"- output written: {self.file_written}",
                f"- data quality: {statuses}",
            ]
        )


def _read_csv(path: Path, columns: list[str], *, required: bool) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"GPU cloud input file not found: {path}")
        return pd.DataFrame(columns=columns)
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        frame = pd.DataFrame(columns=columns)
    if list(frame.columns) != columns:
        raise ValueError(
            f"CSV schema mismatch for {path}: expected {columns}, "
            f"found {list(frame.columns)}"
        )
    return frame


def _as_bool(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def _clean_inputs(
    history: pd.DataFrame,
    fetch_log: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    warnings: list[str] = []
    clean_history = history.copy()
    clean_log = fetch_log.copy()

    for frame, context in ((clean_history, "history"), (clean_log, "fetch log")):
        frame["snapshot_timestamp_utc"] = pd.to_datetime(
            frame["snapshot_timestamp_utc"], errors="coerce", utc=True
        )
        if frame["snapshot_timestamp_utc"].isna().any():
            raise ValueError(f"GPU cloud {context} has an invalid timestamp")
        frame["snapshot_date"] = frame["snapshot_timestamp_utc"].dt.strftime(
            "%Y-%m-%d"
        )
        frame["provider"] = frame["provider"].fillna("").astype(str)
        frame["pricing_type"] = (
            frame["pricing_type"].fillna("").astype(str)
        )

    if not clean_history.empty:
        clean_history = clean_history.loc[
            clean_history["provider"].eq(VAST_PROVIDER)
        ].copy()
        clean_history["reliability"] = pd.to_numeric(
            clean_history["reliability"], errors="coerce"
        )
        clean_history["price_per_gpu_hour_usd"] = pd.to_numeric(
            clean_history["price_per_gpu_hour_usd"], errors="coerce"
        )
        clean_history["interruptible_price_per_gpu_hour_usd"] = pd.to_numeric(
            clean_history["interruptible_price_per_gpu_hour_usd"],
            errors="coerce",
        )
        clean_history["gpu_count"] = pd.to_numeric(
            clean_history["gpu_count"], errors="coerce"
        )
        clean_history["is_available_clean"] = _as_bool(
            clean_history["is_available"]
        )
        clean_history["is_rentable_clean"] = _as_bool(
            clean_history["is_rentable"]
        )
        clean_history["verified_clean"] = _as_bool(clean_history["verified"])
        clean_history["inventory_verifiable_clean"] = _as_bool(
            clean_history["inventory_count_is_verifiable"]
        )
        duplicate_key = [
            "snapshot_timestamp_utc",
            "provider",
            "offer_id",
            "pricing_type",
        ]
        if clean_history.duplicated(duplicate_key).any():
            warnings.append("schema_warning: duplicate history keys were ignored")
            clean_history = clean_history.drop_duplicates(
                duplicate_key, keep="last"
            )

    clean_log = clean_log.loc[clean_log["provider"].eq(VAST_PROVIDER)].copy()
    clean_log["status"] = (
        clean_log["status"].fillna("").astype(str).str.strip().str.upper()
    )
    clean_log["offer_count"] = pd.to_numeric(
        clean_log["offer_count"], errors="coerce"
    )
    clean_log["request_count"] = pd.to_numeric(
        clean_log["request_count"], errors="coerce"
    )
    duplicate_log_key = [
        "snapshot_timestamp_utc",
        "provider",
        "pricing_type",
    ]
    if clean_log.duplicated(duplicate_log_key).any():
        warnings.append("schema_warning: duplicate fetch-log keys were ignored")
        clean_log = clean_log.drop_duplicates(duplicate_log_key, keep="last")
    clean_log = clean_log.sort_values(
        ["snapshot_timestamp_utc", "pricing_type"], kind="stable"
    ).reset_index(drop=True)
    return clean_history, clean_log, warnings


def _latest_attempt_rows(fetch_log: pd.DataFrame) -> pd.DataFrame:
    return (
        fetch_log.sort_values("snapshot_timestamp_utc", kind="stable")
        .drop_duplicates(["snapshot_date", "pricing_type"], keep="last")
        .sort_values(["snapshot_date", "pricing_type"], kind="stable")
        .reset_index(drop=True)
    )


def select_daily_eligible_snapshots(
    fetch_log: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Select each date/pricing type's maximum eligible UTC timestamp."""

    eligible_mask = fetch_log.apply(
        is_snapshot_eligible_for_signals,
        axis=1,
    )
    eligible = fetch_log.loc[eligible_mask].copy()
    if eligible.empty:
        return eligible, 0
    selected = (
        eligible.sort_values("snapshot_timestamp_utc", kind="stable")
        .drop_duplicates(["snapshot_date", "pricing_type"], keep="last")
        .sort_values(["snapshot_date", "pricing_type"], kind="stable")
        .reset_index(drop=True)
    )
    legacy_partial_count = int(
        selected["status"].eq(LEGACY_PARTIAL).sum()
    )
    return selected, legacy_partial_count


def _log_for(
    latest_log: pd.DataFrame,
    snapshot_date: str,
    pricing_type: str,
) -> pd.Series | None:
    rows = latest_log.loc[
        latest_log["snapshot_date"].eq(snapshot_date)
        & latest_log["pricing_type"].eq(pricing_type)
    ]
    return None if rows.empty else rows.iloc[-1]


def _offers_for_log(
    history: pd.DataFrame,
    log_row: pd.Series | None,
    gpu_model: str,
) -> pd.DataFrame:
    if log_row is None or not is_snapshot_eligible_for_signals(log_row):
        return history.iloc[0:0].copy()
    timestamp = log_row["snapshot_timestamp_utc"]
    pricing_type = str(log_row["pricing_type"])
    rows = history.loc[
        history["snapshot_timestamp_utc"].eq(timestamp)
        & history["pricing_type"].eq(pricing_type)
        & history["gpu_model"].eq(gpu_model)
    ].copy()
    if rows.empty:
        return rows
    return rows.loc[
        rows["is_available_clean"]
        & rows["is_rentable_clean"]
        & rows["verified_clean"]
        & rows["reliability"].ge(VAST_MIN_RELIABILITY)
    ].copy()


def _valid_prices(rows: pd.DataFrame, column: str) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=float)
    values = pd.to_numeric(rows[column], errors="coerce")
    return values.loc[values.notna() & values.gt(0) & values.map(math.isfinite)]


def _trend_reference(
    current_date: date,
    current_value: float | int | None,
    daily_values: dict[date, float | int],
    horizon_days: int,
) -> tuple[float | None, date | None]:
    """Return a natural-day change using the nearest allowed daily snapshot."""

    if current_value is None:
        return None, None
    target = current_date - timedelta(days=horizon_days)
    candidates = [
        candidate
        for candidate in daily_values
        if candidate < current_date
        and abs((candidate - target).days) <= TREND_REFERENCE_TOLERANCE_DAYS
    ]
    if not candidates:
        return None, None
    reference_date = min(
        candidates,
        key=lambda candidate: (abs((candidate - target).days), candidate),
    )
    reference_value = daily_values[reference_date]
    if reference_value <= 0:
        return None, None
    return current_value / reference_value - 1.0, reference_date


def classify_supply_signal(
    *,
    rental_price_trend_7d: float | None,
    rental_price_trend_30d: float | None,
    visible_offer_count_trend_7d: float | None,
    visible_offer_count_trend_30d: float | None,
    data_available: bool,
) -> str:
    """Classify Vast.ai's marginal public marketplace supply condition."""

    if not data_available:
        return "DATA_UNAVAILABLE"
    if (
        rental_price_trend_30d is None
        or visible_offer_count_trend_30d is None
    ):
        return "INSUFFICIENT_HISTORY"
    if (
        rental_price_trend_30d <= -0.10
        and visible_offer_count_trend_30d >= 0.20
        and (
            (
                rental_price_trend_7d is not None
                and rental_price_trend_7d < 0
            )
            or (
                visible_offer_count_trend_7d is not None
                and visible_offer_count_trend_7d > 0
            )
        )
    ):
        return "OVERSUPPLY_WARNING"
    if (
        abs(rental_price_trend_30d) < 0.05
        and abs(visible_offer_count_trend_30d) < 0.10
    ):
        return "STABLE"
    if (
        rental_price_trend_30d < 0
        and visible_offer_count_trend_30d > 0
    ):
        return "LOOSENING"
    if (
        rental_price_trend_30d > 0
        and visible_offer_count_trend_30d < 0
    ):
        return "TIGHTENING"
    return "MIXED"


def _status_from_context(
    selected: pd.Series | None,
    latest_attempt: pd.Series | None,
) -> str:
    context = selected if selected is not None else latest_attempt
    return PROVIDER_ERROR if context is None else str(context["status"])


def _notes_from_log(log_row: pd.Series | None) -> list[str]:
    if log_row is None:
        return ["provider_error: missing request status"]
    value = log_row.get("data_quality_notes")
    if pd.isna(value) or not str(value).strip():
        return []
    return [str(value).strip()]


def _timestamp_iso(log_row: pd.Series | None) -> str | None:
    if log_row is None:
        return None
    timestamp = pd.Timestamp(log_row["snapshot_timestamp_utc"])
    return timestamp.isoformat().replace("+00:00", "Z")


def _data_quality(
    *,
    on_demand_snapshot: pd.Series | None,
    interruptible_snapshot: pd.Series | None,
    latest_on_demand_attempt: pd.Series | None,
    latest_interruptible_attempt: pd.Series | None,
    on_demand_rows: pd.DataFrame,
    on_demand_prices: pd.Series,
    missing_trends: list[int],
    notes: list[str],
) -> tuple[str, str]:
    on_context = (
        on_demand_snapshot
        if on_demand_snapshot is not None
        else latest_on_demand_attempt
    )
    on_status = (
        PROVIDER_ERROR if on_context is None else str(on_context["status"])
    )
    bid_context = (
        interruptible_snapshot
        if interruptible_snapshot is not None
        else latest_interruptible_attempt
    )
    bid_status = (
        PROVIDER_ERROR if bid_context is None else str(bid_context["status"])
    )

    if on_demand_snapshot is None and on_status == API_KEY_MISSING:
        status = "API_KEY_MISSING"
    elif on_demand_snapshot is None and on_status == SCHEMA_ERROR:
        status = "SCHEMA_ERROR"
    elif on_demand_snapshot is None:
        status = "PROVIDER_ERROR"
    elif on_status == NO_MARKET_DATA:
        status = "NO_MARKET_DATA"
    elif any("schema_warning" in note for note in notes):
        status = "SCHEMA_WARNING"
    elif on_demand_rows.empty:
        status = "NO_MARKET_DATA"
    elif on_demand_prices.empty:
        status = "SCHEMA_WARNING"
        notes.append("schema_warning: no valid on-demand price")
    elif interruptible_snapshot is None or bid_status == NO_MARKET_DATA:
        status = "PARTIAL_DAY"
        notes.append(
            "interruptible snapshot unavailable for the selected daily date"
        )
    elif missing_trends:
        status = "INSUFFICIENT_HISTORY"
        notes.append(
            "missing natural-day trend reference(s): "
            + ", ".join(f"{horizon}d" for horizon in missing_trends)
            + f" within +/-{TREND_REFERENCE_TOLERANCE_DAYS} days"
        )
    else:
        status = "OK"
    if status not in DATA_QUALITY_STATUSES:
        raise AssertionError(f"Unsupported data quality status: {status}")
    return status, "; ".join(normalize_snapshot_warnings(notes))


def build_gpu_cloud_market_signals(
    history: pd.DataFrame,
    fetch_log: pd.DataFrame,
) -> pd.DataFrame:
    """Rebuild the complete daily signal history from raw offer snapshots."""

    history, fetch_log, global_warnings = _clean_inputs(history, fetch_log)
    if fetch_log.empty:
        raise ValueError("GPU cloud fetch log has no Vast.ai request rows")
    selected_log, legacy_partial_count = select_daily_eligible_snapshots(
        fetch_log
    )
    latest_attempts = _latest_attempt_rows(fetch_log)
    snapshot_dates = sorted(latest_attempts["snapshot_date"].unique())
    if legacy_partial_count:
        compatibility_note = (
            "legacy_partial_compatibility: "
            f"{legacy_partial_count} snapshot(s) treated as "
            "SUCCESS_WITH_WARNINGS"
        )
        global_warnings.append(compatibility_note)
        LOGGER.info(compatibility_note)

    daily_on_demand_prices: dict[str, dict[date, float]] = {
        model: {} for model in TRACKED_GPU_MODELS
    }
    daily_visible_offer_counts: dict[str, dict[date, int]] = {
        model: {} for model in TRACKED_GPU_MODELS
    }
    for snapshot_date in snapshot_dates:
        on_log = _log_for(selected_log, snapshot_date, "on_demand")
        if on_log is None:
            continue
        for model in TRACKED_GPU_MODELS:
            rows = _offers_for_log(history, on_log, model)
            prices = _valid_prices(rows, "price_per_gpu_hour_usd")
            daily_visible_offer_counts[model][
                date.fromisoformat(snapshot_date)
            ] = len(rows)
            if not prices.empty:
                daily_on_demand_prices[model][
                    date.fromisoformat(snapshot_date)
                ] = float(prices.median())

    output_rows: list[dict[str, object]] = []
    for snapshot_date in snapshot_dates:
        current_date = date.fromisoformat(snapshot_date)
        on_log = _log_for(selected_log, snapshot_date, "on_demand")
        bid_log = _log_for(selected_log, snapshot_date, "interruptible")
        latest_on_attempt = _log_for(
            latest_attempts, snapshot_date, "on_demand"
        )
        latest_bid_attempt = _log_for(
            latest_attempts, snapshot_date, "interruptible"
        )
        queried_successfully = on_log is not None
        on_source_timestamp = _timestamp_iso(on_log)
        bid_source_timestamp = _timestamp_iso(bid_log)
        source_timestamp = on_source_timestamp or bid_source_timestamp
        on_snapshot_status = _status_from_context(on_log, latest_on_attempt)
        bid_snapshot_status = _status_from_context(bid_log, latest_bid_attempt)
        snapshot_status = on_snapshot_status

        for model in TRACKED_GPU_MODELS:
            on_rows = _offers_for_log(history, on_log, model)
            bid_rows = _offers_for_log(history, bid_log, model)
            on_prices = _valid_prices(on_rows, "price_per_gpu_hour_usd")
            bid_prices = _valid_prices(
                bid_rows, "interruptible_price_per_gpu_hour_usd"
            )
            on_median = None if on_prices.empty else float(on_prices.median())
            bid_median = None if bid_prices.empty else float(bid_prices.median())
            trend_7d, reference_7d = _trend_reference(
                current_date,
                on_median,
                daily_on_demand_prices[model],
                7,
            )
            trend_30d, reference_30d = _trend_reference(
                current_date,
                on_median,
                daily_on_demand_prices[model],
                30,
            )
            current_offer_count = len(on_rows) if queried_successfully else None
            offer_trend_7d, offer_reference_7d = _trend_reference(
                current_date,
                current_offer_count,
                daily_visible_offer_counts[model],
                7,
            )
            offer_trend_30d, offer_reference_30d = _trend_reference(
                current_date,
                current_offer_count,
                daily_visible_offer_counts[model],
                30,
            )
            missing_trends = [
                horizon
                for horizon, trend in ((7, trend_7d), (30, trend_30d))
                if trend is None
            ]

            provider_available: bool | None
            if queried_successfully:
                provider_available = not on_rows.empty
            else:
                provider_available = None
            visible_gpu_count = (
                on_rows.loc[
                    on_rows["inventory_verifiable_clean"], "gpu_count"
                ].sum(min_count=1)
                if not on_rows.empty
                else 0
            )
            if pd.isna(visible_gpu_count):
                visible_gpu_count = None
            elif float(visible_gpu_count).is_integer():
                visible_gpu_count = int(visible_gpu_count)

            notes = [*global_warnings]
            notes.extend(
                _notes_from_log(on_log if on_log is not None else latest_on_attempt)
            )
            notes.extend(
                _notes_from_log(
                    bid_log if bid_log is not None else latest_bid_attempt
                )
            )
            if not on_rows.empty and not on_rows[
                "inventory_verifiable_clean"
            ].all():
                notes.append(
                    "schema_warning: visible_gpu_count excludes offer(s) "
                    "with unverifiable num_gpus"
                )
            if reference_7d is not None:
                notes.append(f"7d trend reference date: {reference_7d.isoformat()}")
            if reference_30d is not None:
                notes.append(
                    f"30d trend reference date: {reference_30d.isoformat()}"
                )
            if offer_reference_7d is not None:
                notes.append(
                    "7d offer-count trend reference date: "
                    f"{offer_reference_7d.isoformat()}"
                )
            if offer_reference_30d is not None:
                notes.append(
                    "30d offer-count trend reference date: "
                    f"{offer_reference_30d.isoformat()}"
                )
            data_quality_status, data_quality_notes = _data_quality(
                on_demand_snapshot=on_log,
                interruptible_snapshot=bid_log,
                latest_on_demand_attempt=latest_on_attempt,
                latest_interruptible_attempt=latest_bid_attempt,
                on_demand_rows=on_rows,
                on_demand_prices=on_prices,
                missing_trends=missing_trends,
                notes=notes,
            )
            interruptible_discount = (
                1.0 - bid_median / on_median
                if on_median is not None and bid_median is not None
                else None
            )
            supply_signal = classify_supply_signal(
                rental_price_trend_7d=trend_7d,
                rental_price_trend_30d=trend_30d,
                visible_offer_count_trend_7d=offer_trend_7d,
                visible_offer_count_trend_30d=offer_trend_30d,
                data_available=queried_successfully,
            )
            output_rows.append(
                {
                    "date": snapshot_date,
                    "gpu_model": model,
                    "provider": VAST_PROVIDER,
                    "source_snapshot_timestamp_utc": source_timestamp,
                    "on_demand_source_snapshot_timestamp_utc": (
                        on_source_timestamp
                    ),
                    "interruptible_source_snapshot_timestamp_utc": (
                        bid_source_timestamp
                    ),
                    "snapshot_status": snapshot_status,
                    "on_demand_snapshot_status": on_snapshot_status,
                    "interruptible_snapshot_status": bid_snapshot_status,
                    "daily_snapshot_selection_method": (
                        DAILY_SELECTION_METHOD
                    ),
                    "on_demand_median_price_per_gpu_hour": on_median,
                    "on_demand_p25_price_per_gpu_hour": (
                        None if on_prices.empty else float(on_prices.quantile(0.25))
                    ),
                    "on_demand_p10_price_per_gpu_hour": (
                        None if on_prices.empty else float(on_prices.quantile(0.10))
                    ),
                    "interruptible_median_price_per_gpu_hour": bid_median,
                    "rental_price_trend_7d": trend_7d,
                    "rental_price_trend_30d": trend_30d,
                    "visible_offer_count_trend_7d": offer_trend_7d,
                    "visible_offer_count_trend_30d": offer_trend_30d,
                    "visible_offer_count": (
                        current_offer_count
                    ),
                    "visible_gpu_count": (
                        visible_gpu_count if queried_successfully else None
                    ),
                    "supply_signal": supply_signal,
                    "interruptible_discount": interruptible_discount,
                    "provider_available": provider_available,
                    "configured_provider_count": 1,
                    "providers_queried_successfully": (
                        1 if queried_successfully else 0
                    ),
                    "providers_available": (
                        int(provider_available)
                        if provider_available is not None
                        else None
                    ),
                    "cross_provider_availability": (
                        float(provider_available)
                        if provider_available is not None
                        else None
                    ),
                    "availability_scope": AVAILABILITY_SCOPE,
                    "inventory_scope": INVENTORY_SCOPE,
                    "data_quality_status": data_quality_status,
                    "data_quality_notes": data_quality_notes,
                }
            )

    output = pd.DataFrame(output_rows, columns=GPU_CLOUD_SIGNAL_COLUMNS)
    return output.sort_values(
        ["date", "gpu_model", "provider"], kind="stable"
    ).reset_index(drop=True)


def validate_gpu_cloud_market_signals(signals: pd.DataFrame) -> None:
    if list(signals.columns) != GPU_CLOUD_SIGNAL_COLUMNS:
        raise ValueError("GPU cloud signal schema is not stable")
    if signals.empty:
        raise ValueError("GPU cloud signals cannot be empty")
    if signals.duplicated(["date", "gpu_model", "provider"]).any():
        raise ValueError("GPU cloud signals contain duplicate natural keys")
    if not signals["daily_snapshot_selection_method"].eq(
        DAILY_SELECTION_METHOD
    ).all():
        raise ValueError("GPU cloud signals contain an invalid selection method")
    if not set(signals["data_quality_status"]).issubset(
        DATA_QUALITY_STATUSES
    ):
        raise ValueError("GPU cloud signals contain an invalid quality status")
    if not set(signals["supply_signal"]).issubset(SUPPLY_SIGNAL_STATUSES):
        raise ValueError("GPU cloud signals contain an invalid supply signal")
    expected_models = set(TRACKED_GPU_MODELS)
    for snapshot_date, rows in signals.groupby("date", sort=False):
        if set(rows["gpu_model"]) != expected_models:
            raise ValueError(
                f"GPU cloud signals for {snapshot_date} do not contain the "
                "configured GPU universe"
            )


def _canonical_csv(signals: pd.DataFrame) -> str:
    validate_gpu_cloud_market_signals(signals)
    buffer = StringIO()
    signals.to_csv(
        buffer,
        columns=GPU_CLOUD_SIGNAL_COLUMNS,
        index=False,
        encoding="utf-8",
        float_format=CSV_FLOAT_FORMAT,
        na_rep="",
        lineterminator="\n",
    )
    return buffer.getvalue()


def write_gpu_cloud_market_signals(
    signals: pd.DataFrame,
    output_file: Path | str,
) -> bool:
    path = Path(output_file)
    expected_content = _canonical_csv(signals)
    if path.exists() and path.read_text(encoding="utf-8") == expected_content:
        return False

    def validate_temp_file(temp_path: Path) -> None:
        if temp_path.read_text(encoding="utf-8") != expected_content:
            raise ValueError(
                f"Temporary GPU cloud signal verification failed: {temp_path}"
            )

    atomic_write_csv(
        signals,
        path,
        GPU_CLOUD_SIGNAL_COLUMNS,
        float_format=CSV_FLOAT_FORMAT,
        na_rep="",
        lineterminator="\n",
        validate_temp_file=validate_temp_file,
    )
    return True


def run_gpu_cloud_market_signals_update(
    *,
    history_file: Path | str = GPU_CLOUD_HISTORY_FILE,
    fetch_log_file: Path | str = GPU_CLOUD_FETCH_LOG_FILE,
    output_file: Path | str = GPU_CLOUD_SIGNALS_FILE,
) -> GPUCloudSignalsSummary:
    history = _read_csv(
        Path(history_file), GPU_CLOUD_HISTORY_COLUMNS, required=False
    )
    fetch_log = _read_csv(
        Path(fetch_log_file), GPU_CLOUD_FETCH_LOG_COLUMNS, required=True
    )
    signals = build_gpu_cloud_market_signals(history, fetch_log)
    output_path = Path(output_file)
    written = write_gpu_cloud_market_signals(signals, output_path)
    status_counts = {
        str(status): int(count)
        for status, count in signals["data_quality_status"].value_counts().items()
    }
    return GPUCloudSignalsSummary(
        output_file=output_path,
        rows=len(signals),
        dates=int(signals["date"].nunique()),
        latest_date=str(signals["date"].max()),
        file_written=written,
        status_counts=status_counts,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        summary = run_gpu_cloud_market_signals_update()
    except Exception as error:
        print(
            "GPU cloud market signal build failed: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    print(summary.format())


if __name__ == "__main__":
    main()

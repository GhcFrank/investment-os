"""Collect a durable Vast.ai GPU market snapshot using Search Offers only."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from market_data.gpu_cloud_config import (
    GPU_CLOUD_FETCH_LOG_COLUMNS,
    GPU_CLOUD_FETCH_LOG_FILE,
    GPU_CLOUD_FETCH_LOG_KEY,
    GPU_CLOUD_HISTORY_COLUMNS,
    GPU_CLOUD_HISTORY_FILE,
    GPU_CLOUD_HISTORY_KEY,
    VAST_PROVIDER,
    VAST_QUERY_GPU_NAMES,
    VAST_SEARCH_OFFERS_ENDPOINT,
)
from market_data.gpu_cloud_status import (
    NO_MARKET_DATA,
    SUCCESS,
    SUCCESS_WITH_WARNINGS,
    determine_snapshot_status,
)
from market_data.vast_ai_client import (
    VastAIAuthenticationError,
    VastAIClient,
    VastAIError,
    VastAISchemaError,
    VastSearchResult,
    normalize_vast_offers,
)
from utils.csv_utils import atomic_write_csv
from utils.env_utils import get_project_environment_value


LOGGER = logging.getLogger(__name__)
PRICING_TYPES = ("on_demand", "interruptible")
CSV_FLOAT_FORMAT = "%.15g"


@dataclass(frozen=True)
class GPUCloudMarketUpdateSummary:
    snapshot_timestamp_utc: str
    history_file: Path
    fetch_log_file: Path
    offers_collected: int
    pricing_types_succeeded: int
    pricing_types_failed: int
    requests_made: int
    history_file_written: bool
    fetch_log_file_written: bool
    warnings: tuple[str, ...] = ()

    def format(self) -> str:
        return "\n".join(
            [
                "GPU cloud market update summary:",
                f"- provider: {VAST_PROVIDER}",
                f"- snapshot: {self.snapshot_timestamp_utc}",
                f"- offers collected: {self.offers_collected}",
                (
                    "- pricing types succeeded/failed: "
                    f"{self.pricing_types_succeeded}/"
                    f"{self.pricing_types_failed}"
                ),
                f"- Search Offers requests: {self.requests_made}",
                f"- history written: {self.history_file_written}",
                f"- fetch log written: {self.fetch_log_file_written}",
                f"- warnings: {len(self.warnings)}",
            ]
        )


def _snapshot_time(value: datetime | None) -> datetime:
    timestamp = value or datetime.now(UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC).replace(second=0, microsecond=0)


def _utc_iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _read_existing(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return _empty(columns)
    try:
        existing = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return _empty(columns)
    if list(existing.columns) != columns:
        raise ValueError(
            f"CSV schema mismatch for {path}: expected {columns}, "
            f"found {list(existing.columns)}"
        )
    return existing


def _canonical_frame(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    key_columns: list[str],
) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column not in output:
            output[column] = pd.NA
    output = output.reindex(columns=columns)
    for column in key_columns:
        output[column] = output[column].fillna("").astype(str).str.strip()
        if output[column].eq("").any():
            raise ValueError(f"GPU cloud CSV contains a blank key: {column}")
    numeric_columns = {
        "gpu_count",
        "instance_price_per_hour_usd",
        "price_per_gpu_hour_usd",
        "reliability",
        "min_bid_price_per_hour_usd",
        "interruptible_price_per_gpu_hour_usd",
        "offer_count",
        "request_count",
    }
    for column in numeric_columns.intersection(output.columns):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output = output.drop_duplicates(subset=key_columns, keep="last")
    return output.sort_values(key_columns, kind="stable").reset_index(drop=True)


def _csv_content(frame: pd.DataFrame, columns: list[str]) -> str:
    buffer = StringIO()
    frame.to_csv(
        buffer,
        columns=columns,
        index=False,
        encoding="utf-8",
        float_format=CSV_FLOAT_FORMAT,
        na_rep="",
        lineterminator="\n",
    )
    return buffer.getvalue()


def _upsert_csv(
    incoming: pd.DataFrame,
    path: Path,
    *,
    columns: list[str],
    key_columns: list[str],
) -> bool:
    existing = _read_existing(path, columns)
    combined = _canonical_frame(
        pd.concat([existing, incoming], ignore_index=True),
        columns=columns,
        key_columns=key_columns,
    )
    expected_content = _csv_content(combined, columns)
    if path.exists() and path.read_text(encoding="utf-8") == expected_content:
        return False

    def validate_temp_file(temp_path: Path) -> None:
        if temp_path.read_text(encoding="utf-8") != expected_content:
            raise ValueError(
                f"Temporary GPU cloud CSV verification failed: {temp_path}"
            )

    atomic_write_csv(
        combined,
        path,
        columns,
        float_format=CSV_FLOAT_FORMAT,
        na_rep="",
        lineterminator="\n",
        validate_temp_file=validate_temp_file,
    )
    return True


def _fetch_log_row(
    *,
    timestamp: datetime,
    pricing_type: str,
    status: str,
    offer_count: int,
    request_count: int,
    results_truncated: bool,
    notes: str,
) -> dict[str, Any]:
    timestamp_utc = _utc_iso(timestamp)
    return {
        "snapshot_timestamp_utc": timestamp_utc,
        "snapshot_date": timestamp_utc[:10],
        "provider": VAST_PROVIDER,
        "pricing_type": pricing_type,
        "status": status,
        "offer_count": offer_count,
        "request_count": request_count,
        "results_truncated": results_truncated,
        "data_quality_notes": notes,
        "source_endpoint": VAST_SEARCH_OFFERS_ENDPOINT,
        "ingested_at_utc": timestamp_utc,
    }


def _write_missing_key_log(
    timestamp: datetime,
    fetch_log_file: Path,
) -> None:
    decision = determine_snapshot_status(
        api_key_configured=False,
        request_succeeded=False,
        schema_valid=False,
        valid_offer_count=0,
    )
    rows = [
        _fetch_log_row(
            timestamp=timestamp,
            pricing_type=pricing_type,
            status=decision.status,
            offer_count=0,
            request_count=0,
            results_truncated=False,
            notes=decision.data_quality_notes,
        )
        for pricing_type in PRICING_TYPES
    ]
    _upsert_csv(
        pd.DataFrame(rows, columns=GPU_CLOUD_FETCH_LOG_COLUMNS),
        fetch_log_file,
        columns=GPU_CLOUD_FETCH_LOG_COLUMNS,
        key_columns=GPU_CLOUD_FETCH_LOG_KEY,
    )


def run_gpu_cloud_market_update(
    *,
    client: VastAIClient | Any | None = None,
    history_file: Path | str = GPU_CLOUD_HISTORY_FILE,
    fetch_log_file: Path | str = GPU_CLOUD_FETCH_LOG_FILE,
    now: datetime | None = None,
    load_environment: bool = True,
    env_file: Path | str | None = None,
) -> GPUCloudMarketUpdateSummary:
    """Collect on-demand and interruptible offers and atomically upsert them."""

    timestamp = _snapshot_time(now)
    output_history = Path(history_file)
    output_fetch_log = Path(fetch_log_file)
    if client is None:
        api_key = get_project_environment_value(
            "VAST_API_KEY",
            env_file=env_file,
            load_environment=load_environment,
        )
        if not api_key:
            _write_missing_key_log(timestamp, output_fetch_log)
            raise VastAIAuthenticationError("VAST_API_KEY is not configured")
        client = VastAIClient(api_key)

    frames: list[pd.DataFrame] = []
    fetch_rows: list[dict[str, Any]] = []
    all_warnings: list[str] = []
    failures = 0
    successes = 0
    requests_made = 0

    for pricing_type in PRICING_TYPES:
        try:
            result: VastSearchResult = client.search_offers(
                pricing_type=pricing_type,
                gpu_names=VAST_QUERY_GPU_NAMES,
            )
            requests_made += result.request_count
            normalized, normalization_warnings = normalize_vast_offers(
                result.offers,
                pricing_type=pricing_type,
                snapshot_timestamp=timestamp,
                ingested_at=timestamp,
            )
            warnings = tuple(
                (*result.warnings, *normalization_warnings)
            )
            if not normalized.empty:
                frames.append(normalized)
            decision = determine_snapshot_status(
                api_key_configured=True,
                request_succeeded=True,
                schema_valid=True,
                valid_offer_count=len(normalized),
                source_offer_count=len(result.offers),
                warnings=warnings,
                results_truncated=result.results_truncated,
            )
            all_warnings.extend(decision.warnings)
            fetch_rows.append(
                _fetch_log_row(
                    timestamp=timestamp,
                    pricing_type=pricing_type,
                    status=decision.status,
                    offer_count=len(normalized),
                    request_count=result.request_count,
                    results_truncated=result.results_truncated,
                    notes=decision.data_quality_notes,
                )
            )
            if decision.status in {
                SUCCESS,
                SUCCESS_WITH_WARNINGS,
                NO_MARKET_DATA,
            }:
                successes += 1
            else:
                failures += 1
        except Exception as error:
            failures += 1
            request_count = (
                error.request_count if isinstance(error, VastAIError) else 0
            )
            schema_error = isinstance(error, VastAISchemaError)
            warning_prefix = "schema_error" if schema_error else "provider_error"
            safe_warning = f"{warning_prefix}: {type(error).__name__}"
            decision = determine_snapshot_status(
                api_key_configured=True,
                request_succeeded=schema_error,
                schema_valid=not schema_error,
                valid_offer_count=0,
                warnings=[safe_warning],
            )
            all_warnings.extend(decision.warnings)
            requests_made += request_count
            fetch_rows.append(
                _fetch_log_row(
                    timestamp=timestamp,
                    pricing_type=pricing_type,
                    status=decision.status,
                    offer_count=0,
                    request_count=request_count,
                    results_truncated=False,
                    notes=decision.data_quality_notes,
                )
            )
            LOGGER.error(
                "Vast.ai %s Search Offers failed: %s",
                pricing_type,
                type(error).__name__,
            )

    fetch_log_written = _upsert_csv(
        pd.DataFrame(fetch_rows, columns=GPU_CLOUD_FETCH_LOG_COLUMNS),
        output_fetch_log,
        columns=GPU_CLOUD_FETCH_LOG_COLUMNS,
        key_columns=GPU_CLOUD_FETCH_LOG_KEY,
    )
    history_written = False
    offers_collected = sum(len(frame) for frame in frames)
    if frames:
        history_written = _upsert_csv(
            pd.concat(frames, ignore_index=True),
            output_history,
            columns=GPU_CLOUD_HISTORY_COLUMNS,
            key_columns=GPU_CLOUD_HISTORY_KEY,
        )

    summary = GPUCloudMarketUpdateSummary(
        snapshot_timestamp_utc=_utc_iso(timestamp),
        history_file=output_history,
        fetch_log_file=output_fetch_log,
        offers_collected=offers_collected,
        pricing_types_succeeded=successes,
        pricing_types_failed=failures,
        requests_made=requests_made,
        history_file_written=history_written,
        fetch_log_file_written=fetch_log_written,
        warnings=tuple(sorted(set(all_warnings))),
    )
    if successes == 0:
        raise RuntimeError(
            "No complete Vast.ai Search Offers snapshot was collected; "
            "request outcomes and any returned raw offers were preserved"
        )
    return summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        summary = run_gpu_cloud_market_update()
    except Exception as error:
        print(
            "GPU cloud market update failed: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    print(summary.format())


if __name__ == "__main__":
    main()

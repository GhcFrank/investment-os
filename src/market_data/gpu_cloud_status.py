"""Central snapshot-status rules shared by collection and signal building."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


SUCCESS = "SUCCESS"
SUCCESS_WITH_WARNINGS = "SUCCESS_WITH_WARNINGS"
API_KEY_MISSING = "API_KEY_MISSING"
PROVIDER_ERROR = "PROVIDER_ERROR"
NO_MARKET_DATA = "NO_MARKET_DATA"
SCHEMA_ERROR = "SCHEMA_ERROR"
LEGACY_PARTIAL = "PARTIAL"

SNAPSHOT_STATUSES = frozenset(
    {
        SUCCESS,
        SUCCESS_WITH_WARNINGS,
        API_KEY_MISSING,
        PROVIDER_ERROR,
        NO_MARKET_DATA,
        SCHEMA_ERROR,
    }
)
VALID_SIGNAL_STATUSES = frozenset({SUCCESS, SUCCESS_WITH_WARNINGS})
DAILY_SELECTION_METHOD = "latest_eligible_snapshot_per_pricing_type"


@dataclass(frozen=True)
class SnapshotStatusDecision:
    status: str
    warnings: tuple[str, ...]
    data_quality_notes: str


def normalize_snapshot_warnings(warnings: Iterable[object]) -> tuple[str, ...]:
    """Deduplicate warnings and return a deterministic lexical ordering."""

    return tuple(
        sorted(
            {
                " ".join(str(warning).strip().split())
                for warning in warnings
                if warning is not None and str(warning).strip()
            }
        )
    )


def determine_snapshot_status(
    *,
    api_key_configured: bool,
    request_succeeded: bool,
    schema_valid: bool,
    valid_offer_count: int,
    warnings: Iterable[object] = (),
    results_truncated: bool = False,
    source_offer_count: int | None = None,
) -> SnapshotStatusDecision:
    """Apply the one authoritative status priority for a provider snapshot."""

    if valid_offer_count < 0:
        raise ValueError("valid_offer_count cannot be negative")
    if source_offer_count is not None and source_offer_count < 0:
        raise ValueError("source_offer_count cannot be negative")

    normalized = list(normalize_snapshot_warnings(warnings))
    if not api_key_configured:
        status = API_KEY_MISSING
        normalized = list(
            normalize_snapshot_warnings(
                [*normalized, "VAST_API_KEY is not configured"]
            )
        )
    elif not request_succeeded:
        status = PROVIDER_ERROR
        if not normalized:
            normalized = ["provider_error: Search Offers request failed"]
    elif not schema_valid:
        status = SCHEMA_ERROR
        if not normalized:
            normalized = ["schema_error: invalid Search Offers response"]
    elif results_truncated or any(
        warning.startswith("result_limit_warning:") for warning in normalized
    ):
        status = PROVIDER_ERROR
        normalized = list(
            normalize_snapshot_warnings(
                [
                    *normalized,
                    "provider_error: Search Offers result was truncated",
                ]
            )
        )
    elif (
        source_offer_count is not None
        and source_offer_count > 0
        and valid_offer_count == 0
    ):
        status = SCHEMA_ERROR
        normalized = list(
            normalize_snapshot_warnings(
                [
                    *normalized,
                    "schema_error: no source offers could be normalized",
                ]
            )
        )
    elif valid_offer_count == 0:
        status = NO_MARKET_DATA
    elif normalized:
        status = SUCCESS_WITH_WARNINGS
    else:
        status = SUCCESS

    return SnapshotStatusDecision(
        status=status,
        warnings=tuple(normalized),
        data_quality_notes="; ".join(normalized),
    )


def _integer(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return int(parsed) if parsed.is_integer() else None


def _boolean(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def is_snapshot_eligible_for_signals(row: Mapping[str, Any]) -> bool:
    """
    Return whether a fetch-log row is a real daily market observation.

    ``NO_MARKET_DATA`` is eligible only as a verified zero-availability
    observation. Legacy ``PARTIAL`` remains temporarily eligible under the
    strict compatibility conditions requested for existing history.
    """

    status = str(row.get("status", "")).strip().upper()
    request_count = _integer(row.get("request_count"))
    offer_count = _integer(row.get("offer_count"))
    results_truncated = _boolean(row.get("results_truncated"))
    complete = results_truncated is False

    if status in VALID_SIGNAL_STATUSES:
        return bool(
            request_count is not None
            and request_count > 0
            and offer_count is not None
            and offer_count > 0
            and complete
        )
    if status == NO_MARKET_DATA:
        return bool(
            request_count is not None
            and request_count > 0
            and offer_count == 0
            and complete
        )
    if status == LEGACY_PARTIAL:
        return bool(
            request_count is not None
            and request_count > 0
            and offer_count is not None
            and offer_count > 0
            and complete
        )
    return False


def is_successful_request_snapshot(row: Mapping[str, Any]) -> bool:
    """Return whether the eligible row represents a completed API query."""

    return is_snapshot_eligible_for_signals(row)

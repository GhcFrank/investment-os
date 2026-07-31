"""Read-only Vast.ai Search Offers client and response normalization."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import requests

from market_data.gpu_cloud_config import (
    GPU_NAME_ALIASES,
    GPU_CLOUD_HISTORY_COLUMNS,
    VAST_MAX_ATTEMPTS,
    VAST_MIN_RELIABILITY,
    VAST_ONLY_RENTABLE,
    VAST_ONLY_VERIFIED,
    VAST_PROVIDER,
    VAST_REQUEST_TIMEOUT_SECONDS,
    VAST_RETRY_BASE_SECONDS,
    VAST_SEARCH_LIMIT,
    VAST_SEARCH_OFFERS_ENDPOINT,
)


class VastAIError(RuntimeError):
    """Base error for safe, secret-free Vast.ai failures."""

    def __init__(self, message: str, *, request_count: int = 0) -> None:
        super().__init__(message)
        self.request_count = request_count


class VastAIAuthenticationError(VastAIError):
    """The API key is absent or rejected."""


class VastAIRateLimitError(VastAIError):
    """The Search Offers rate limit remained active after retries."""


class VastAIResponseError(VastAIError):
    """The endpoint returned an HTTP or JSON response we cannot use."""


class VastAISchemaError(VastAIError):
    """The JSON response does not have the documented offers structure."""


@dataclass(frozen=True)
class VastSearchResult:
    offers: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]
    request_count: int
    results_truncated: bool


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _positive_number(value: object) -> float | None:
    parsed = _number(value)
    return parsed if parsed is not None and parsed > 0 else None


def _boolean(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return None


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return (
        value.astimezone(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def canonicalize_gpu_model(
    gpu_name_raw: object,
    gpu_ram_megabytes: object = None,
) -> tuple[str, str | None]:
    """Return a conservative canonical model and an optional warning."""

    raw = " ".join(str(gpu_name_raw or "").strip().upper().split())
    if raw in GPU_NAME_ALIASES:
        return GPU_NAME_ALIASES[raw], None

    if raw in {"A100 PCIE", "A100 SXM4", "A100 SXM"}:
        gpu_ram = _number(gpu_ram_megabytes)
        if gpu_ram is not None and gpu_ram >= 80_000:
            return "A100_80GB", None
        fallback = re.sub(r"[^A-Z0-9]+", "_", raw).strip("_")
        return (
            f"{fallback}_UNKNOWN_MEMORY",
            f"gpu_model_warning: {raw} memory is not reliably 80GB",
        )

    if raw == "H200":
        return (
            "H200",
            "gpu_model_warning: Vast.ai H200 form factor is not specified; "
            "not classified as H200_SXM",
        )

    safe = re.sub(r"[^A-Z0-9]+", "_", raw).strip("_") or "UNKNOWN"
    return safe, f"gpu_model_warning: unrecognized Vast.ai GPU name {raw!r}"


def _location_fields(value: object) -> tuple[str, str]:
    region = " ".join(str(value or "").strip().split())
    if not region:
        return "", ""
    country = region.rsplit(",", maxsplit=1)[-1].strip().upper()
    if not re.fullmatch(r"[A-Z]{2}", country):
        country = ""
    return region, country


class VastAIClient:
    """Minimal client restricted to the documented Search Offers endpoint."""

    def __init__(
        self,
        api_key: str,
        *,
        session: requests.Session | Any | None = None,
        timeout_seconds: int = VAST_REQUEST_TIMEOUT_SECONDS,
        max_attempts: int = VAST_MAX_ATTEMPTS,
        retry_base_seconds: float = VAST_RETRY_BASE_SECONDS,
        sleep_func: Callable[[float], None] = time.sleep,
    ) -> None:
        if not str(api_key or "").strip():
            raise VastAIAuthenticationError("VAST_API_KEY is not configured")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._api_key = str(api_key).strip()
        self._session = session or requests.Session()
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._retry_base_seconds = retry_base_seconds
        self._sleep = sleep_func

    def _request_offers(
        self,
        payload: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], int]:
        """POST one Search Offers request with bounded retry/backoff."""

        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._session.post(
                    VAST_SEARCH_OFFERS_ENDPOINT,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                    timeout=self._timeout_seconds,
                )
            except (requests.Timeout, requests.ConnectionError) as error:
                if attempt == self._max_attempts:
                    raise VastAIResponseError(
                        "Vast.ai Search Offers network request failed after "
                        f"{attempt} attempt(s): {type(error).__name__}",
                        request_count=attempt,
                    ) from None
                self._sleep(self._retry_base_seconds * (2 ** (attempt - 1)))
                continue

            status_code = int(response.status_code)
            if status_code in {401, 403}:
                raise VastAIAuthenticationError(
                    f"Vast.ai Search Offers rejected credentials (HTTP "
                    f"{status_code})",
                    request_count=attempt,
                )
            if status_code == 429 or 500 <= status_code < 600:
                if attempt == self._max_attempts:
                    error_type = (
                        VastAIRateLimitError
                        if status_code == 429
                        else VastAIResponseError
                    )
                    raise error_type(
                        "Vast.ai Search Offers failed after "
                        f"{attempt} attempt(s) (HTTP {status_code})",
                        request_count=attempt,
                    )
                self._sleep(self._retry_base_seconds * (2 ** (attempt - 1)))
                continue
            if status_code < 200 or status_code >= 300:
                raise VastAIResponseError(
                    f"Vast.ai Search Offers returned HTTP {status_code}",
                    request_count=attempt,
                )

            try:
                body = response.json()
            except (TypeError, ValueError):
                raise VastAIResponseError(
                    "Vast.ai Search Offers returned invalid JSON",
                    request_count=attempt,
                ) from None
            if not isinstance(body, dict) or not isinstance(
                body.get("offers"), list
            ):
                raise VastAISchemaError(
                    "Vast.ai Search Offers response must contain an offers list",
                    request_count=attempt,
                )
            if not all(isinstance(offer, dict) for offer in body["offers"]):
                raise VastAISchemaError(
                    "Vast.ai Search Offers contains a non-object offer",
                    request_count=attempt,
                )
            return body["offers"], attempt

        raise AssertionError("unreachable Vast.ai retry state")

    def search_offers(
        self,
        *,
        pricing_type: str,
        gpu_names: Sequence[str],
        limit: int = VAST_SEARCH_LIMIT,
        min_reliability: float = VAST_MIN_RELIABILITY,
        only_verified: bool = VAST_ONLY_VERIFIED,
        only_rentable: bool = VAST_ONLY_RENTABLE,
    ) -> VastSearchResult:
        """
        Fetch offers using documented filters and adaptive query partitioning.

        Search Offers does not document a page cursor. If a multi-GPU query
        reaches ``limit``, the client splits the configured GPU names into
        smaller queries. A saturated single-name query is retained but marked
        truncated instead of inventing an unsupported pagination parameter.
        """

        if pricing_type not in {"on_demand", "interruptible"}:
            raise ValueError("pricing_type must be on_demand or interruptible")
        names = tuple(dict.fromkeys(str(name).strip() for name in gpu_names))
        if not names or any(not name for name in names):
            raise ValueError("gpu_names must contain non-empty names")
        if limit < 1:
            raise ValueError("limit must be positive")

        pending: list[tuple[str, ...]] = [names]
        offers_by_id: dict[str, dict[str, Any]] = {}
        warnings: list[str] = []
        request_count = 0
        results_truncated = False
        api_type = "ondemand" if pricing_type == "on_demand" else "bid"

        while pending:
            batch = pending.pop(0)
            payload: dict[str, Any] = {
                "limit": limit,
                "type": api_type,
                "gpu_name": {"in": list(batch)},
                "reliability": {"gte": min_reliability},
                "order": [["dph_total", "asc"]],
            }
            if only_verified:
                payload["verified"] = {"eq": True}
            if only_rentable:
                payload["rentable"] = {"eq": True}
                payload["rented"] = {"eq": False}

            try:
                batch_offers, batch_request_count = self._request_offers(
                    payload
                )
            except VastAIError as error:
                error.request_count += request_count
                raise
            request_count += batch_request_count
            if len(batch_offers) >= limit and len(batch) > 1:
                midpoint = len(batch) // 2
                pending[0:0] = [batch[:midpoint], batch[midpoint:]]
                continue
            if len(batch_offers) >= limit:
                results_truncated = True
                warnings.append(
                    "result_limit_warning: Search Offers reached limit for "
                    f"gpu_name={batch[0]!r}; visible inventory may be partial"
                )

            for offer in batch_offers:
                offer_id = offer.get("id")
                if offer_id is None or str(offer_id).strip() == "":
                    warnings.append(
                        "schema_warning: skipped offer without id"
                    )
                    continue
                offers_by_id[str(offer_id)] = offer

        return VastSearchResult(
            offers=tuple(offers_by_id.values()),
            warnings=tuple(dict.fromkeys(warnings)),
            request_count=request_count,
            results_truncated=results_truncated,
        )


def normalize_vast_offers(
    offers: Iterable[dict[str, Any]],
    *,
    pricing_type: str,
    snapshot_timestamp: datetime,
    ingested_at: datetime | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Normalize Search Offers rows without inventing missing source values."""

    if pricing_type not in {"on_demand", "interruptible"}:
        raise ValueError("pricing_type must be on_demand or interruptible")
    snapshot_timestamp_utc = _utc_iso(snapshot_timestamp)
    snapshot_date = snapshot_timestamp_utc[:10]
    ingested_at_utc = _utc_iso(ingested_at or snapshot_timestamp)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for offer in offers:
        offer_id = offer.get("id")
        gpu_name_raw = " ".join(str(offer.get("gpu_name") or "").split())
        if offer_id is None or str(offer_id).strip() == "":
            warnings.append("schema_warning: skipped offer without id")
            continue
        if not gpu_name_raw:
            warnings.append(
                f"schema_warning: skipped offer {offer_id} without gpu_name"
            )
            continue

        gpu_count_value = _positive_number(offer.get("num_gpus"))
        gpu_count: int | float | None = None
        if gpu_count_value is not None:
            gpu_count = (
                int(gpu_count_value)
                if gpu_count_value.is_integer()
                else gpu_count_value
            )
        else:
            warnings.append(
                f"schema_warning: offer {offer_id} has invalid num_gpus"
            )

        instance_price = _positive_number(offer.get("dph_total"))
        if instance_price is None:
            warnings.append(
                f"schema_warning: offer {offer_id} has invalid dph_total"
            )
        price_per_gpu = (
            instance_price / gpu_count_value
            if instance_price is not None and gpu_count_value is not None
            else None
        )
        min_bid = _positive_number(offer.get("min_bid"))
        gpu_model, model_warning = canonicalize_gpu_model(
            gpu_name_raw,
            offer.get("gpu_ram"),
        )
        if model_warning:
            warnings.append(model_warning)
        region, country = _location_fields(offer.get("geolocation"))
        rentable = _boolean(offer.get("rentable"))
        rented = _boolean(offer.get("rented"))
        verification = str(offer.get("verification") or "").strip().lower()
        verified: bool | None = verification == "verified" if verification else None
        if verified is None and "verified" in offer:
            verified = _boolean(offer.get("verified"))
        if rentable is None:
            warnings.append(
                f"schema_warning: offer {offer_id} has invalid rentable"
            )
        if rented is None:
            warnings.append(
                f"schema_warning: offer {offer_id} has invalid rented"
            )
        if verified is None:
            warnings.append(
                f"schema_warning: offer {offer_id} has no verification state"
            )
        is_available = (
            rentable and not rented
            if rentable is not None and rented is not None
            else None
        )
        reliability = _number(
            offer.get("reliability2", offer.get("reliability"))
        )
        if reliability is None:
            warnings.append(
                f"schema_warning: offer {offer_id} has invalid reliability"
            )

        rows.append(
            {
                "snapshot_timestamp_utc": snapshot_timestamp_utc,
                "snapshot_date": snapshot_date,
                "provider": VAST_PROVIDER,
                "offer_id": offer_id,
                "machine_id": offer.get("machine_id"),
                "host_id": offer.get("host_id"),
                "region": region,
                "country": country,
                "gpu_name_raw": gpu_name_raw,
                "gpu_model": gpu_model,
                "gpu_count": gpu_count,
                "pricing_type": pricing_type,
                "instance_price_per_hour_usd": instance_price,
                "price_per_gpu_hour_usd": price_per_gpu,
                "is_available": is_available,
                "is_rentable": rentable,
                "verified": verified,
                "reliability": reliability,
                "min_bid_price_per_hour_usd": min_bid,
                "interruptible_price_per_gpu_hour_usd": (
                    price_per_gpu
                    if pricing_type == "interruptible"
                    else None
                ),
                "inventory_count_is_verifiable": gpu_count is not None,
                "source_endpoint": VAST_SEARCH_OFFERS_ENDPOINT,
                "ingested_at_utc": ingested_at_utc,
            }
        )

    normalized = pd.DataFrame(rows, columns=GPU_CLOUD_HISTORY_COLUMNS)
    if not normalized.empty:
        normalized = normalized.sort_values(
            ["snapshot_timestamp_utc", "provider", "pricing_type", "offer_id"],
            kind="stable",
            key=lambda column: column.astype(str),
        ).reset_index(drop=True)
    return normalized, tuple(dict.fromkeys(warnings))

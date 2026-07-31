"""Configuration and stable schemas for GPU cloud market collection.

The CSV ``provider`` field is a stable source identifier. ``vast_ai`` means
the row was derived directly from Vast.ai Search Offers, not from a broker or
another cloud provider.
"""

from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

VAST_PROVIDER = "vast_ai"
VAST_API_BASE_URL = "https://console.vast.ai/api/v0"
VAST_SEARCH_OFFERS_ENDPOINT = f"{VAST_API_BASE_URL}/bundles"
VAST_REQUEST_TIMEOUT_SECONDS = 20
VAST_MAX_ATTEMPTS = 3
VAST_RETRY_BASE_SECONDS = 1.0
VAST_SEARCH_LIMIT = 1000
VAST_MIN_RELIABILITY = 0.95
VAST_ONLY_VERIFIED = True
VAST_ONLY_RENTABLE = True

TRACKED_GPU_MODELS = (
    "H100_SXM",
    "H100_PCIE",
    "H200_SXM",
    "A100_80GB",
    "L40S",
    "B200",
    "RTX_4090",
    "RTX_5090",
)

GPU_NAME_ALIASES = {
    "H100 SXM": "H100_SXM",
    "H100SXM": "H100_SXM",
    "H100 PCIE": "H100_PCIE",
    "H100 PCI-E": "H100_PCIE",
    "H200 SXM": "H200_SXM",
    "A100 80GB": "A100_80GB",
    "A100 SXM4 80GB": "A100_80GB",
    "A100 PCIE 80GB": "A100_80GB",
    "L40S": "L40S",
    "B200": "B200",
    "RTX 4090": "RTX_4090",
    "RTX 5090": "RTX_5090",
}

# These are the current Vast.ai Search Offers names that can be queried
# without guessing at a model's identity. A100 memory is resolved from the
# per-offer gpu_ram field during normalization. Vast currently exposes H200 as
# a generic name; it is retained as H200 rather than mislabeled H200_SXM.
VAST_QUERY_GPU_NAMES = (
    "H100 SXM",
    "H100 PCIE",
    "H200",
    "A100 SXM4",
    "A100 PCIE",
    "L40S",
    "B200",
    "RTX 4090",
    "RTX 5090",
)

GPU_CLOUD_HISTORY_FILE = (
    BASE_DIR / "data" / "market_data" / "gpu_cloud_market_history.csv"
)
GPU_CLOUD_FETCH_LOG_FILE = (
    BASE_DIR / "data" / "market_data" / "gpu_cloud_market_fetch_log.csv"
)
GPU_CLOUD_SIGNALS_FILE = (
    BASE_DIR / "data" / "signals" / "gpu_cloud_market_signals.csv"
)

GPU_CLOUD_HISTORY_COLUMNS = [
    "snapshot_timestamp_utc",
    "snapshot_date",
    "provider",
    "offer_id",
    "machine_id",
    "host_id",
    "region",
    "country",
    "gpu_name_raw",
    "gpu_model",
    "gpu_count",
    "pricing_type",
    "instance_price_per_hour_usd",
    "price_per_gpu_hour_usd",
    "is_available",
    "is_rentable",
    "verified",
    "reliability",
    "min_bid_price_per_hour_usd",
    "interruptible_price_per_gpu_hour_usd",
    "inventory_count_is_verifiable",
    "source_endpoint",
    "ingested_at_utc",
]

GPU_CLOUD_HISTORY_KEY = [
    "snapshot_timestamp_utc",
    "provider",
    "offer_id",
    "pricing_type",
]

GPU_CLOUD_FETCH_LOG_COLUMNS = [
    "snapshot_timestamp_utc",
    "snapshot_date",
    "provider",
    "pricing_type",
    "status",
    "offer_count",
    "request_count",
    "results_truncated",
    "data_quality_notes",
    "source_endpoint",
    "ingested_at_utc",
]

GPU_CLOUD_FETCH_LOG_KEY = [
    "snapshot_timestamp_utc",
    "provider",
    "pricing_type",
]

GPU_CLOUD_SIGNAL_COLUMNS = [
    "date",
    "gpu_model",
    "provider",
    "source_snapshot_timestamp_utc",
    "on_demand_source_snapshot_timestamp_utc",
    "interruptible_source_snapshot_timestamp_utc",
    "snapshot_status",
    "on_demand_snapshot_status",
    "interruptible_snapshot_status",
    "daily_snapshot_selection_method",
    "on_demand_median_price_per_gpu_hour",
    "on_demand_p25_price_per_gpu_hour",
    "on_demand_p10_price_per_gpu_hour",
    "interruptible_median_price_per_gpu_hour",
    "rental_price_trend_7d",
    "rental_price_trend_30d",
    "visible_offer_count_trend_7d",
    "visible_offer_count_trend_30d",
    "visible_offer_count",
    "visible_gpu_count",
    "supply_signal",
    "interruptible_discount",
    "provider_available",
    "configured_provider_count",
    "providers_queried_successfully",
    "providers_available",
    "cross_provider_availability",
    "availability_scope",
    "inventory_scope",
    "data_quality_status",
    "data_quality_notes",
]

AVAILABILITY_SCOPE = "Vast.ai only; not yet cross-provider"
INVENTORY_SCOPE = (
    "Visible Vast.ai offers matching configured filters at the latest "
    "snapshot; not total provider capacity"
)

SUPPLY_SIGNAL_SCOPE = (
    "Vast.ai marginal public marketplace indicator; not the whole GPU cloud "
    "market"
)

SUPPLY_SIGNAL_STATUSES = {
    "INSUFFICIENT_HISTORY",
    "TIGHTENING",
    "STABLE",
    "LOOSENING",
    "OVERSUPPLY_WARNING",
    "MIXED",
    "DATA_UNAVAILABLE",
}

TREND_HORIZONS_DAYS = (7, 30)
TREND_REFERENCE_TOLERANCE_DAYS = 2

DATA_QUALITY_STATUSES = {
    "OK",
    "PARTIAL",
    "PARTIAL_DAY",
    "INSUFFICIENT_HISTORY",
    "API_KEY_MISSING",
    "PROVIDER_ERROR",
    "NO_MARKET_DATA",
    "SCHEMA_ERROR",
    "SCHEMA_WARNING",
}

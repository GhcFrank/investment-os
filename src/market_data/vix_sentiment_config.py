"""Stable schema and paths for the formal daily VIX sentiment signal."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
VIX_MARKET_SENTIMENT_SIGNAL_FILE = (
    BASE_DIR / "data" / "signals" / "vix_market_sentiment.csv"
)

VIX_MARKET_SENTIMENT_SIGNAL_COLUMNS = [
    "date",
    "vix",
    "change_1d",
    "change_5d",
    "change_20d",
    "sentiment_regime",
    "signal",
    "source",
    "source_status",
    "data_status",
    "stale",
    "updated_at",
    "data_quality_notes",
]

VIX_DATA_STATUSES = {
    "SUCCESS",
    "SUCCESS_WITH_WARNINGS",
    "INSUFFICIENT_HISTORY",
    "DATA_UNAVAILABLE",
}

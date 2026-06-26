"""
Update daily VIX and CNN Fear & Greed market sentiment indicators.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf

from utils.date_utils import today_et_str


BASE_DIR = Path(__file__).resolve().parents[2]
MARKET_DATA_DIR = BASE_DIR / "data" / "market_data"

VIX_CURRENT_FILE = MARKET_DATA_DIR / "vix.csv"
VIX_HISTORY_FILE = MARKET_DATA_DIR / "vix_history.csv"
CNN_CURRENT_FILE = MARKET_DATA_DIR / "cnn_fear_greed.csv"
CNN_HISTORY_FILE = MARKET_DATA_DIR / "cnn_fear_greed_history.csv"
SENTIMENT_FETCH_LOG_FILE = MARKET_DATA_DIR / "sentiment_fetch_log.csv"

VIX_COLUMNS = [
    "date",
    "vix",
    "level",
    "source",
    "status",
    "error_message",
    "change_1d",
    "change_5d",
    "change_20d",
    "updated_at",
]

CNN_COLUMNS = [
    "date",
    "fear_greed_index",
    "level",
    "source",
    "status",
    "error_message",
    "change_1d",
    "change_5d",
    "change_20d",
    "updated_at",
]

FETCH_LOG_COLUMNS = [
    "run_at",
    "status",
    "vix_status",
    "cnn_status",
    "rows_written",
    "error_message",
]

CNN_DATED_ENDPOINT = (
    "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{date}"
)
CNN_ENDPOINT = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"


def now_utc_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)

    try:
        df = pd.read_csv(path, dtype=str)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)

    for column in columns:
        if column not in df.columns:
            df[column] = ""

    return df.reindex(columns=columns)


def write_csv(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = df.copy()

    for column in columns:
        if column not in output.columns:
            output[column] = ""

    temp_path = path.with_name(f".{path.name}.tmp")
    output.reindex(columns=columns).to_csv(temp_path, index=False, encoding="utf-8")
    temp_path.replace(path)


def format_number(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""

    if pd.isna(numeric):
        return ""

    return f"{numeric:.2f}"


def vix_level(value: float) -> str:
    if value < 15:
        return "calm"
    if value < 20:
        return "normal"
    if value < 30:
        return "risk_off"
    return "stress"


def cnn_level(value: float) -> str:
    if value < 25:
        return "extreme_fear"
    if value < 45:
        return "fear"
    if value <= 55:
        return "neutral"
    if value <= 75:
        return "greed"
    return "extreme_greed"


def summarize_cnn_error(error: Exception | str) -> str:
    """
    Return a short, user-readable error message for CNN Fear & Greed failures.
    """

    text = str(error)
    lowered = text.lower()
    prefix = "CNN Fear & Greed unavailable:"

    for status_code in ["418", "403"]:
        if f"http {status_code}" in lowered or f"{status_code} client error" in lowered:
            return f"{prefix} HTTP {status_code} from CNN graphdata endpoint"

    if "timeout" in lowered or "timed out" in lowered:
        return f"{prefix} request timeout"

    json_terms = ["json", "decode", "invalid response", "not a json"]
    if any(term in lowered for term in json_terms):
        return f"{prefix} invalid response format"

    reason = re.sub(r"https?://\S+", "", text)
    reason = re.sub(r"\s+", " ", reason).strip(" ;:")
    if not reason:
        reason = "unknown error"

    message = f"{prefix} {reason}"
    return message[:160]


def normalize_rating(value: object) -> str:
    text = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    return text or "unknown"


def vix_row(
    *,
    date: str,
    value: object,
    level: str,
    source: str = "yfinance",
    status: str = "ok",
    error_message: str = "",
    updated_at: str | None = None,
) -> dict[str, str]:
    return {
        "date": date,
        "vix": format_number(value),
        "level": level,
        "source": source,
        "status": status,
        "error_message": error_message,
        "change_1d": "",
        "change_5d": "",
        "change_20d": "",
        "updated_at": updated_at or today_et_str(),
    }


def cnn_row(
    *,
    date: str,
    value: object,
    level: str,
    source: str = "cnn_unofficial",
    status: str = "ok",
    error_message: str = "",
    updated_at: str | None = None,
) -> dict[str, str]:
    return {
        "date": date,
        "fear_greed_index": format_number(value),
        "level": level,
        "source": source,
        "status": status,
        "error_message": error_message,
        "change_1d": "",
        "change_5d": "",
        "change_20d": "",
        "updated_at": updated_at or today_et_str(),
    }


def fetch_vix(lookback_days: int = 60) -> dict[str, str]:
    period = "6mo" if lookback_days <= 120 else "1y"
    hist = yf.download(
        "^VIX",
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if hist.empty:
        raise RuntimeError("No VIX data returned from yfinance.")

    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

    if "Close" not in hist.columns:
        raise RuntimeError("VIX data does not contain a Close column.")

    close = hist["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = pd.to_numeric(close, errors="coerce").dropna()
    if close.empty:
        raise RuntimeError("VIX Close series has no valid values.")

    latest_value = float(close.iloc[-1])
    latest_index = close.index[-1]
    latest_date = pd.Timestamp(latest_index).date().isoformat()

    return vix_row(
        date=latest_date,
        value=latest_value,
        level=vix_level(latest_value),
    )


def date_from_timestamp(value: object) -> str:
    if value is None or value == "":
        return ""

    if isinstance(value, (int, float)):
        seconds = float(value) / 1000 if float(value) > 10_000_000_000 else float(value)
        return datetime.fromtimestamp(seconds, tz=UTC).date().isoformat()

    text = str(value).strip()
    if not text:
        return ""

    try:
        numeric = float(text)
        return date_from_timestamp(numeric)
    except ValueError:
        pass

    try:
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        return datetime.fromisoformat(normalized).date().isoformat()
    except ValueError:
        return ""


def extract_historical_cnn_value(payload: dict[str, Any]) -> dict[str, Any] | None:
    historical = payload.get("fear_and_greed_historical", {})
    data = historical.get("data") if isinstance(historical, dict) else None

    if not isinstance(data, list) or not data:
        return None

    for item in reversed(data):
        if not isinstance(item, dict):
            continue

        value = item.get("score", item.get("value", item.get("y")))
        if value is None:
            continue

        return {
            "score": value,
            "rating": item.get("rating", ""),
            "timestamp": item.get("timestamp", item.get("x", item.get("date", ""))),
        }

    return None


def parse_cnn_payload(payload: dict[str, Any], date_hint: str | None = None) -> dict[str, str]:
    current = payload.get("fear_and_greed", {})

    if not isinstance(current, dict):
        current = {}

    score = current.get("score")
    rating = current.get("rating", "")
    timestamp = current.get("timestamp", "")

    if score is None:
        fallback = extract_historical_cnn_value(payload)
        if fallback is not None:
            score = fallback.get("score")
            rating = fallback.get("rating", "")
            timestamp = fallback.get("timestamp", "")

    try:
        value = float(score)
    except (TypeError, ValueError):
        raise ValueError("CNN payload does not contain a valid Fear & Greed score.")

    level = normalize_rating(rating)
    if level == "unknown":
        level = cnn_level(value)

    date = date_from_timestamp(timestamp) or date_hint or today_et_str()

    return cnn_row(
        date=date,
        value=value,
        level=level,
    )


def failed_cnn_row(error_message: Exception | str) -> dict[str, str]:
    return cnn_row(
        date=today_et_str(),
        value="",
        level="unknown",
        status="failed",
        error_message=summarize_cnn_error(error_message),
    )


def fetch_cnn_fear_greed() -> dict[str, str]:
    today = today_et_str()
    user_agent = os.getenv("SENTIMENT_USER_AGENT", "InvestmentOS-SentimentBot/1.0")
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json,text/plain,*/*",
    }
    errors: list[str] = []

    for url in [CNN_DATED_ENDPOINT.format(date=today), CNN_ENDPOINT]:
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("CNN response is not a JSON object.")
            return parse_cnn_payload(payload, date_hint=today)
        except Exception as error:
            detail = f"{url}: {error}"
            errors.append(detail)
            print(f"Warning: CNN Fear & Greed fetch failed: {detail}", file=sys.stderr)

    return failed_cnn_row("; ".join(errors))


def numeric_value(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if pd.isna(numeric):
        return None

    return numeric


def add_change_columns(
    rows: pd.DataFrame,
    existing_history: pd.DataFrame,
    value_column: str,
    columns: list[str],
) -> pd.DataFrame:
    updated_rows = rows.copy()
    current_dates = set(updated_rows["date"].dropna().astype(str))
    history = existing_history[
        ~existing_history["date"].fillna("").astype(str).isin(current_dates)
    ].copy()
    combined = pd.concat([history, updated_rows], ignore_index=True)
    combined["_numeric_value"] = combined[value_column].map(numeric_value)

    for row_index, row in updated_rows.iterrows():
        if row.get("status") != "ok" or numeric_value(row.get(value_column)) is None:
            continue

        indicator_history = combined[
            (combined["status"] == "ok") & combined["_numeric_value"].notna()
        ].copy()
        indicator_history = indicator_history.sort_values(
            by=["date", "updated_at"],
            kind="stable",
        ).reset_index(drop=True)
        matches = indicator_history[indicator_history["date"] == row["date"]]

        if matches.empty:
            continue

        position = matches.index[-1]
        latest_value = float(indicator_history.loc[position, "_numeric_value"])

        for days in [1, 5, 20]:
            previous_position = position - days
            if previous_position < 0:
                continue

            previous_value = float(
                indicator_history.loc[previous_position, "_numeric_value"]
            )
            updated_rows.loc[row_index, f"change_{days}d"] = format_number(
                latest_value - previous_value
            )

    return updated_rows.reindex(columns=columns)


def upsert_history(
    existing_history: pd.DataFrame,
    rows: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    combined = pd.concat([existing_history, rows], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"], keep="last")
    combined = combined.sort_values(by=["date"], kind="stable")
    combined = combined.reset_index(drop=True)
    return combined.reindex(columns=columns)


def current_snapshot(history: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=columns)

    snapshot = history.sort_values(by=["date"], kind="stable").tail(1)
    return snapshot.reindex(columns=columns)


def append_fetch_log(row: dict[str, object]) -> None:
    existing = read_csv(SENTIMENT_FETCH_LOG_FILE, FETCH_LOG_COLUMNS)
    combined = pd.concat(
        [existing, pd.DataFrame([row], columns=FETCH_LOG_COLUMNS)],
        ignore_index=True,
    )
    write_csv(combined, SENTIMENT_FETCH_LOG_FILE, FETCH_LOG_COLUMNS)


def _skipped_cnn_row() -> dict[str, str]:
    return cnn_row(
        date=today_et_str(),
        value="",
        level="unknown",
        status="skipped",
        error_message="Skipped by --skip-cnn.",
    )


def _build_outputs(
    vix_data: dict[str, str],
    cnn_data: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    existing_vix = read_csv(VIX_HISTORY_FILE, VIX_COLUMNS)
    existing_cnn = read_csv(CNN_HISTORY_FILE, CNN_COLUMNS)

    vix_rows = pd.DataFrame([vix_data], columns=VIX_COLUMNS)
    cnn_rows = pd.DataFrame([cnn_data], columns=CNN_COLUMNS)
    vix_rows = add_change_columns(vix_rows, existing_vix, "vix", VIX_COLUMNS)
    cnn_rows = add_change_columns(
        cnn_rows,
        existing_cnn,
        "fear_greed_index",
        CNN_COLUMNS,
    )
    vix_history = upsert_history(existing_vix, vix_rows, VIX_COLUMNS)
    cnn_history = upsert_history(existing_cnn, cnn_rows, CNN_COLUMNS)

    return vix_rows, cnn_rows, vix_history, cnn_history


def update_sentiment_indicators(
    *,
    dry_run: bool = False,
    skip_cnn: bool = False,
    lookback_days: int = 60,
) -> tuple[int, dict[str, pd.DataFrame]]:
    run_at = now_utc_iso()

    try:
        vix_data = fetch_vix(lookback_days=lookback_days)
    except Exception as error:
        error_message = str(error)[:160]
        if not dry_run:
            append_fetch_log(
                {
                    "run_at": run_at,
                    "status": "failed",
                    "vix_status": "failed",
                    "cnn_status": "skipped",
                    "rows_written": 0,
                    "error_message": error_message,
                }
            )
        print(f"Error fetching VIX: {error_message}", file=sys.stderr)
        return 1, {}

    cnn_data = _skipped_cnn_row() if skip_cnn else fetch_cnn_fear_greed()
    vix_rows, cnn_rows, vix_history, cnn_history = _build_outputs(vix_data, cnn_data)

    if dry_run:
        print("Dry run: sentiment indicators were not written.")
        print("\nVIX")
        print(vix_rows.to_string(index=False))
        print("\nCNN Fear & Greed")
        print(cnn_rows.to_string(index=False))
        return 0, {"vix": vix_rows, "cnn": cnn_rows}

    write_csv(vix_history, VIX_HISTORY_FILE, VIX_COLUMNS)
    write_csv(current_snapshot(vix_history, VIX_COLUMNS), VIX_CURRENT_FILE, VIX_COLUMNS)
    write_csv(cnn_history, CNN_HISTORY_FILE, CNN_COLUMNS)
    write_csv(current_snapshot(cnn_history, CNN_COLUMNS), CNN_CURRENT_FILE, CNN_COLUMNS)

    cnn_error = cnn_data["error_message"] if cnn_data["status"] == "failed" else ""
    append_fetch_log(
        {
            "run_at": run_at,
            "status": "success",
            "vix_status": vix_data["status"],
            "cnn_status": cnn_data["status"],
            "rows_written": len(vix_rows) + len(cnn_rows),
            "error_message": cnn_error,
        }
    )

    print("Updated sentiment indicators.")
    print("\nVIX")
    print(vix_rows.to_string(index=False))
    print("\nCNN Fear & Greed")
    print(cnn_rows.to_string(index=False))
    return 0, {"vix": vix_rows, "cnn": cnn_rows}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update VIX and CNN Fear & Greed market sentiment indicators."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--lookback-days", type=int, default=60)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    exit_code, _ = update_sentiment_indicators(
        dry_run=args.dry_run,
        skip_cnn=args.skip_cnn,
        lookback_days=args.lookback_days,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

"""
Update daily market sentiment indicators.
"""

from __future__ import annotations

import argparse
import os
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

SENTIMENT_CURRENT_FILE = MARKET_DATA_DIR / "sentiment_indicators.csv"
SENTIMENT_HISTORY_FILE = MARKET_DATA_DIR / "sentiment_indicator_history.csv"
SENTIMENT_FETCH_LOG_FILE = MARKET_DATA_DIR / "sentiment_fetch_log.csv"

SENTIMENT_COLUMNS = [
    "date",
    "indicator",
    "value",
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


def normalize_rating(value: object) -> str:
    text = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    return text or "unknown"


def sentiment_row(
    *,
    date: str,
    indicator: str,
    value: object,
    level: str,
    source: str,
    status: str,
    error_message: str = "",
    updated_at: str | None = None,
) -> dict[str, str]:
    return {
        "date": date,
        "indicator": indicator,
        "value": format_number(value),
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

    return sentiment_row(
        date=latest_date,
        indicator="VIX",
        value=latest_value,
        level=vix_level(latest_value),
        source="yfinance",
        status="ok",
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

    return sentiment_row(
        date=date,
        indicator="CNN_FEAR_GREED",
        value=value,
        level=level,
        source="cnn_unofficial",
        status="ok",
    )


def failed_cnn_row(error_message: str) -> dict[str, str]:
    return {
        "date": today_et_str(),
        "indicator": "CNN_FEAR_GREED",
        "value": "",
        "level": "unknown",
        "source": "cnn_unofficial",
        "status": "failed",
        "error_message": error_message[:500],
        "change_1d": "",
        "change_5d": "",
        "change_20d": "",
        "updated_at": today_et_str(),
    }


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
            errors.append(f"{url}: {error}")

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
) -> pd.DataFrame:
    updated_rows = rows.copy()
    current_keys = set(zip(updated_rows["date"], updated_rows["indicator"], strict=False))
    history = existing_history[
        ~existing_history.apply(
            lambda row: (row["date"], row["indicator"]) in current_keys,
            axis=1,
        )
    ].copy()
    combined = pd.concat([history, updated_rows], ignore_index=True)
    combined["_numeric_value"] = combined["value"].map(numeric_value)

    for row_index, row in updated_rows.iterrows():
        if row.get("status") != "ok" or numeric_value(row.get("value")) is None:
            continue

        indicator_history = combined[
            (combined["indicator"] == row["indicator"])
            & (combined["status"] == "ok")
            & combined["_numeric_value"].notna()
        ].copy()
        indicator_history = indicator_history.sort_values(
            by=["date", "updated_at"],
            kind="stable",
        ).reset_index(drop=True)
        matches = indicator_history[
            (indicator_history["date"] == row["date"])
            & (indicator_history["indicator"] == row["indicator"])
        ]

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

    return updated_rows.reindex(columns=SENTIMENT_COLUMNS)


def upsert_history(
    existing_history: pd.DataFrame,
    rows: pd.DataFrame,
) -> pd.DataFrame:
    combined = pd.concat([existing_history, rows], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "indicator"], keep="last")
    combined = combined.sort_values(by=["date", "indicator"], kind="stable")
    combined = combined.reset_index(drop=True)
    return combined.reindex(columns=SENTIMENT_COLUMNS)


def current_snapshot(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=SENTIMENT_COLUMNS)

    snapshot = history.sort_values(by=["date", "indicator"], kind="stable")
    snapshot = snapshot.drop_duplicates(subset=["indicator"], keep="last")
    return snapshot.reindex(columns=SENTIMENT_COLUMNS)


def append_fetch_log(row: dict[str, object]) -> None:
    existing = read_csv(SENTIMENT_FETCH_LOG_FILE, FETCH_LOG_COLUMNS)
    combined = pd.concat(
        [existing, pd.DataFrame([row], columns=FETCH_LOG_COLUMNS)],
        ignore_index=True,
    )
    write_csv(combined, SENTIMENT_FETCH_LOG_FILE, FETCH_LOG_COLUMNS)


def update_sentiment_indicators(
    *,
    dry_run: bool = False,
    skip_cnn: bool = False,
    lookback_days: int = 60,
) -> tuple[int, pd.DataFrame]:
    run_at = now_utc_iso()
    rows_written = 0

    try:
        vix_row = fetch_vix(lookback_days=lookback_days)
    except Exception as error:
        error_message = str(error)
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
        return 1, pd.DataFrame(columns=SENTIMENT_COLUMNS)

    if skip_cnn:
        cnn_row = {
            "date": today_et_str(),
            "indicator": "CNN_FEAR_GREED",
            "value": "",
            "level": "unknown",
            "source": "cnn_unofficial",
            "status": "skipped",
            "error_message": "Skipped by --skip-cnn.",
            "change_1d": "",
            "change_5d": "",
            "change_20d": "",
            "updated_at": today_et_str(),
        }
    else:
        cnn_row = fetch_cnn_fear_greed()

    rows = pd.DataFrame([vix_row, cnn_row], columns=SENTIMENT_COLUMNS)
    existing_history = read_csv(SENTIMENT_HISTORY_FILE, SENTIMENT_COLUMNS)
    rows = add_change_columns(rows, existing_history)
    updated_history = upsert_history(existing_history, rows)
    snapshot = current_snapshot(updated_history)

    if dry_run:
        print("Dry run: sentiment indicators were not written.")
        print(rows.to_string(index=False))
        return 0, rows

    write_csv(updated_history, SENTIMENT_HISTORY_FILE, SENTIMENT_COLUMNS)
    write_csv(snapshot, SENTIMENT_CURRENT_FILE, SENTIMENT_COLUMNS)
    rows_written = len(rows)
    append_fetch_log(
        {
            "run_at": run_at,
            "status": "success",
            "vix_status": vix_row["status"],
            "cnn_status": cnn_row["status"],
            "rows_written": rows_written,
            "error_message": "",
        }
    )

    print("Updated sentiment indicators.")
    print(rows.to_string(index=False))
    return 0, rows


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

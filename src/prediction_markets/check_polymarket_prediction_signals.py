"""
check_polymarket_prediction_signals.py

作用：
1. 读取 Polymarket earnings prediction snapshot/history
2. 检测新增市场、概率变化、临近 earnings date、成交/流动性变化
3. 如果有信号，发送邮件总结
"""

from datetime import date

import pandas as pd

from prediction_markets.polymarket_utils import BASE_DIR, now_et, today_et, to_float
from utils.send_email import send_email


SNAPSHOT_FILE = BASE_DIR / "data" / "prediction_markets" / "polymarket_predictions.csv"
HISTORY_FILE = BASE_DIR / "data" / "prediction_markets" / "polymarket_predictions_history.csv"
EARNINGS_CALENDAR_FILE = BASE_DIR / "data" / "events" / "earnings_calendar.csv"

ONE_DAY_PROB_THRESHOLD = 0.10
FIVE_DAY_PROB_THRESHOLD = 0.20
VOLUME_LIQUIDITY_THRESHOLD = 0.50


def parse_date(value) -> date | None:
    """
    Parse a date-like value.
    """

    if value is None or str(value).strip() == "":
        return None

    parsed = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(parsed):
        return None

    return parsed.date()


def pct(value) -> str:
    """
    Format probability as percent.
    """

    numeric = to_float(value)

    if numeric is None:
        return "n/a"

    return f"{numeric:.0%}"


def pct_point_change(value) -> str:
    """
    Format probability delta as percentage points.
    """

    numeric = to_float(value)

    if numeric is None:
        return "n/a"

    sign = "+" if numeric >= 0 else ""

    return f"{sign}{numeric * 100:.0f} pct"


def latest_earnings_dates() -> dict[str, date]:
    """
    Load latest yfinance earnings date per ticker.
    """

    if not EARNINGS_CALENDAR_FILE.exists():
        return {}

    calendar = pd.read_csv(EARNINGS_CALENDAR_FILE)

    if calendar.empty:
        return {}

    if "date" in calendar.columns:
        latest_snapshot_date = calendar["date"].max()
        calendar = calendar[calendar["date"] == latest_snapshot_date]

    result = {}

    for _, row in calendar.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        earnings_dates = str(row.get("earnings_dates", "")).strip()

        if not ticker or not earnings_dates:
            continue

        first_date = earnings_dates.split(";")[0]
        parsed = parse_date(first_date)

        if parsed is not None:
            result[ticker] = parsed

    return result


def get_previous_snapshot(
    history: pd.DataFrame,
    current_date: str,
    lookback_days: int,
) -> pd.DataFrame:
    """
    Get the latest history rows on or before current_date - lookback_days.
    """

    if history.empty:
        return pd.DataFrame()

    history = history.copy()
    history["date_dt"] = pd.to_datetime(
        history["date"],
        errors="coerce",
    )

    target_date = (
        pd.to_datetime(current_date)
        - pd.Timedelta(days=lookback_days)
    )

    candidates = history[
        history["date_dt"] <= target_date
    ].copy()

    if candidates.empty:
        return pd.DataFrame()

    latest_date = candidates["date_dt"].max()

    return candidates[candidates["date_dt"] == latest_date].drop(columns=["date_dt"])


def merge_previous(
    current: pd.DataFrame,
    previous: pd.DataFrame,
    suffix: str,
) -> pd.DataFrame:
    """
    Attach previous probability/volume/liquidity columns.
    """

    if previous.empty:
        result = current.copy()
        result[f"probability_{suffix}"] = pd.NA
        result[f"volume_{suffix}"] = pd.NA
        result[f"liquidity_{suffix}"] = pd.NA
        return result

    previous = previous[
        [
            "market_id",
            "outcome",
            "probability",
            "volume",
            "liquidity",
        ]
    ].rename(
        columns={
            "probability": f"probability_{suffix}",
            "volume": f"volume_{suffix}",
            "liquidity": f"liquidity_{suffix}",
        }
    )

    return current.merge(
        previous,
        on=["market_id", "outcome"],
        how="left",
    )


def build_signals(
    current: pd.DataFrame,
    history: pd.DataFrame,
) -> list[dict]:
    """
    Build prediction signal rows.
    """

    if current.empty:
        return []

    current_date = today_et()

    prior_history = history[
        history["date"].astype(str) < current_date
    ].copy()

    previous_1d = get_previous_snapshot(
        prior_history,
        current_date,
        1,
    )
    previous_5d = get_previous_snapshot(
        prior_history,
        current_date,
        5,
    )

    merged = merge_previous(current, previous_1d, "1d")
    merged = merge_previous(merged, previous_5d, "5d")

    prior_market_ids = set(prior_history["market_id"].astype(str))
    earnings_dates = latest_earnings_dates()
    current_date_obj = parse_date(current_date)

    signals = []

    for _, row in merged.iterrows():
        market_id = str(row["market_id"])
        ticker = str(row["ticker"]).strip().upper()
        probability = to_float(row.get("probability"))
        probability_1d = to_float(row.get("probability_1d"))
        probability_5d = to_float(row.get("probability_5d"))
        volume = to_float(row.get("volume"))
        volume_1d = to_float(row.get("volume_1d"))
        liquidity = to_float(row.get("liquidity"))
        liquidity_1d = to_float(row.get("liquidity_1d"))

        reasons = []
        change_1d = None
        change_5d = None

        if market_id not in prior_market_ids:
            reasons.append("new market")

        if probability is not None and probability_1d is not None:
            change_1d = probability - probability_1d

            if abs(change_1d) > ONE_DAY_PROB_THRESHOLD:
                reasons.append(f"1d probability change {pct_point_change(change_1d)}")

        if probability is not None and probability_5d is not None:
            change_5d = probability - probability_5d

            if abs(change_5d) > FIVE_DAY_PROB_THRESHOLD:
                reasons.append(f"5d probability change {pct_point_change(change_5d)}")

        earnings_date = earnings_dates.get(ticker)

        if earnings_date is not None and current_date_obj is not None:
            days_to_earnings = (earnings_date - current_date_obj).days

            if 0 <= days_to_earnings <= 3:
                reasons.append(f"earnings in {days_to_earnings} day(s)")

        if volume is not None and volume_1d not in (None, 0):
            volume_change = volume / volume_1d - 1

            if volume_change > VOLUME_LIQUIDITY_THRESHOLD:
                reasons.append(f"volume up {volume_change:.0%}")

        if liquidity is not None and liquidity_1d not in (None, 0):
            liquidity_change = liquidity / liquidity_1d - 1

            if liquidity_change > VOLUME_LIQUIDITY_THRESHOLD:
                reasons.append(f"liquidity up {liquidity_change:.0%}")

        if not reasons:
            continue

        signals.append(
            {
                "ticker": ticker,
                "company": row.get("company", ""),
                "market_id": market_id,
                "question": row.get("question", ""),
                "outcome": row.get("outcome", ""),
                "probability": probability,
                "change_1d": change_1d,
                "change_5d": change_5d,
                "volume": volume,
                "liquidity": liquidity,
                "earnings_date": earnings_date.isoformat()
                if earnings_date is not None
                else "",
                "reasons": "; ".join(reasons),
            }
        )

    return signals


def build_email_body(signals: list[dict]) -> str:
    """
    Build email body.
    """

    lines = [
        "Polymarket Earnings Prediction Summary",
        "",
        f"Generated at: {now_et()}",
        "",
    ]

    for signal in signals:
        lines.append(f"{signal['ticker']} | {signal['company']}")
        lines.append(f"Question: {signal['question']}")
        lines.append(
            f"{signal['outcome']} probability: {pct(signal['probability'])}"
        )
        lines.append(f"1d change: {pct_point_change(signal['change_1d'])}")
        lines.append(f"5d change: {pct_point_change(signal['change_5d'])}")
        lines.append(f"Earnings date: {signal['earnings_date'] or 'n/a'}")
        lines.append(f"Reason: {signal['reasons']}")
        lines.append("")

    lines.append("This alert was generated by Investment OS.")

    return "\n".join(lines)


def main() -> None:
    """
    主程序入口。
    """

    if not SNAPSHOT_FILE.exists():
        print("No Polymarket prediction snapshot found.")
        return

    current = pd.read_csv(SNAPSHOT_FILE)

    if current.empty:
        print("No Polymarket predictions to check.")
        return

    if HISTORY_FILE.exists():
        history = pd.read_csv(HISTORY_FILE)
    else:
        history = pd.DataFrame()

    signals = build_signals(
        current,
        history,
    )

    if not signals:
        print("No Polymarket prediction signals detected.")
        return

    subject = f"Polymarket Earnings Prediction Summary: {len(signals)} signal(s)"
    body = build_email_body(signals)

    send_email(
        subject=subject,
        body=body,
    )

    print(f"Sent Polymarket prediction email with {len(signals)} signal(s).")


if __name__ == "__main__":
    main()

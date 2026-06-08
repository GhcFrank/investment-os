"""
update_polymarket_predictions.py

作用：
1. 读取 data/prediction_markets/polymarket_earnings_watchlist.csv
2. 只跟踪 enabled=True 的 Polymarket markets
3. 输出当前预测快照 data/prediction_markets/polymarket_predictions.csv
4. 追加/更新历史 data/prediction_markets/polymarket_predictions_history.csv
"""

import pandas as pd

from prediction_markets.polymarket_utils import (
    BASE_DIR,
    fetch_gamma_json,
    normalize_bool,
    now_et,
    parse_jsonish_list,
    to_float,
    today_et,
)


WATCHLIST_FILE = BASE_DIR / "data" / "prediction_markets" / "polymarket_earnings_watchlist.csv"
SNAPSHOT_FILE = BASE_DIR / "data" / "prediction_markets" / "polymarket_predictions.csv"
HISTORY_FILE = BASE_DIR / "data" / "prediction_markets" / "polymarket_predictions_history.csv"


OUTPUT_COLUMNS = [
    "date",
    "ticker",
    "company",
    "market_id",
    "question",
    "outcome",
    "probability",
    "volume",
    "liquidity",
    "end_date",
    "updated_at",
]


def fetch_market(market_id: str) -> dict:
    """
    Fetch a single market by id from Polymarket Gamma API.
    """

    data = fetch_gamma_json(f"/markets/{market_id}")

    if not isinstance(data, dict):
        raise ValueError(f"Unexpected market response for market_id={market_id}")

    return data


def market_prediction_rows(
    watchlist_row: pd.Series,
    market: dict,
) -> list[dict]:
    """
    Convert one market into one row per outcome.
    """

    outcomes = parse_jsonish_list(market.get("outcomes"))
    prices = parse_jsonish_list(
        market.get("outcomePrices")
        or market.get("outcome_prices")
    )

    rows = []

    for index, outcome in enumerate(outcomes):
        probability = None

        if index < len(prices):
            probability = to_float(prices[index])

        rows.append(
            {
                "date": today_et(),
                "ticker": str(watchlist_row["ticker"]).strip().upper(),
                "company": watchlist_row.get("company", ""),
                "market_id": market.get("id", watchlist_row["market_id"]),
                "question": market.get("question", watchlist_row.get("question", "")),
                "outcome": outcome,
                "probability": probability,
                "volume": (
                    to_float(market.get("volumeNum"))
                    or to_float(market.get("volume"))
                ),
                "liquidity": (
                    to_float(market.get("liquidityNum"))
                    or to_float(market.get("liquidity"))
                ),
                "end_date": (
                    market.get("endDate")
                    or market.get("endDateIso")
                    or ""
                ),
                "updated_at": now_et(),
            }
        )

    return rows


def load_enabled_watchlist() -> pd.DataFrame:
    """
    Read enabled watchlist rows.
    """

    if not WATCHLIST_FILE.exists():
        return pd.DataFrame()

    watchlist = pd.read_csv(WATCHLIST_FILE)

    if "enabled" not in watchlist.columns:
        raise ValueError("polymarket_earnings_watchlist.csv must contain enabled")

    return watchlist[
        watchlist["enabled"].apply(normalize_bool)
    ].copy()


def save_snapshot_and_history(snapshot_df: pd.DataFrame) -> None:
    """
    Save current snapshot and upsert daily history by date/market/outcome.
    """

    SNAPSHOT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot_df.to_csv(
        SNAPSHOT_FILE,
        index=False,
    )

    if HISTORY_FILE.exists():
        existing = pd.read_csv(HISTORY_FILE)
        combined = pd.concat(
            [existing, snapshot_df],
            ignore_index=True,
        )
    else:
        combined = snapshot_df

    if not combined.empty:
        combined = combined.drop_duplicates(
            subset=["date", "market_id", "outcome"],
            keep="last",
        )

    combined.to_csv(
        HISTORY_FILE,
        index=False,
    )


def main() -> None:
    """
    主程序入口。
    """

    enabled_watchlist = load_enabled_watchlist()

    if enabled_watchlist.empty:
        snapshot_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
        save_snapshot_and_history(snapshot_df)
        print("No enabled Polymarket earnings markets to track.")
        return

    rows = []

    for _, watchlist_row in enabled_watchlist.iterrows():
        market_id = str(watchlist_row["market_id"]).strip()

        if not market_id:
            continue

        print(f"Fetching Polymarket market {market_id}...")

        try:
            market = fetch_market(market_id)
        except Exception as error:
            print(f"Warning: failed to fetch market {market_id}: {error}")
            continue

        rows.extend(
            market_prediction_rows(
                watchlist_row,
                market,
            )
        )

    snapshot_df = pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    )

    save_snapshot_and_history(snapshot_df)

    print(f"\nSaved Polymarket prediction snapshot to: {SNAPSHOT_FILE}")
    print(f"Updated Polymarket prediction history: {HISTORY_FILE}")


if __name__ == "__main__":
    main()

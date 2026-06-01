"""
daily_market_monitor.py

作用：
1. 读取 data/prices.csv
2. 检测每日单股异动
3. 检测主题/板块整体强弱
4. 输出 data/daily_market_signals.csv

这个脚本是 Research OS 的“每日收盘异动监控器”。
"""

from pathlib import Path
from datetime import datetime

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "prices.csv"
OUTPUT_FILE = BASE_DIR / "data" / "daily_market_signals.csv"


def add_stock_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    给每只股票添加单股层面的异动信号。
    """

    df = df.copy()

    # 单日大涨：涨幅 >= 5%
    df["signal_big_up"] = df["daily_return"] >= 0.05

    # 单日大跌：跌幅 <= -5%
    df["signal_big_down"] = df["daily_return"] <= -0.05

    # 放量：今天成交量超过过去20日均量的1.5倍
    df["signal_high_volume"] = df["volume_ratio"] >= 1.5

    # 放量上涨：涨幅 >= 3%，且成交量明显放大
    df["signal_volume_up"] = (
        (df["daily_return"] >= 0.03)
        & (df["volume_ratio"] >= 1.5)
    )

    # 放量下跌：跌幅 <= -3%，且成交量明显放大
    df["signal_volume_down"] = (
        (df["daily_return"] <= -0.03)
        & (df["volume_ratio"] >= 1.5)
    )

    # 接近52周新高：距离52周高点不到5%
    df["signal_near_52w_high"] = df["distance_to_52w_high"] >= -0.05

    # 突破趋势：价格站上20日、50日、200日均线
    df["signal_strong_trend"] = (
        (df["above_20dma"] == True)
        & (df["above_50dma"] == True)
        & (df["above_200dma"] == True)
    )

    return df


def build_theme_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算 theme / subtheme 层面的整体表现。

    例如：
    AI Infrastructure 整体今天涨多少？
    GPU、Networking、Power 哪个最强？
    """

    group_columns = ["theme", "subtheme"]

    rows = []

    for group_col in group_columns:
        if group_col not in df.columns:
            continue

        grouped = df.groupby(group_col)

        for group_name, group_df in grouped:
            rows.append(
                {
                    "signal_date": datetime.now().strftime("%Y-%m-%d"),
                    "signal_type": "group_summary",
                    "group_type": group_col,
                    "group_name": group_name,
                    "ticker": "",
                    "company": "",
                    "daily_return": group_df["daily_return"].mean(),
                    "weekly_return": group_df["weekly_return"].mean(),
                    "monthly_return": group_df["monthly_return"].mean(),
                    "volume_ratio": group_df["volume_ratio"].mean(),
                    "message": (
                        f"{group_col}={group_name}: "
                        f"avg daily return={group_df['daily_return'].mean():.2%}, "
                        f"avg weekly return={group_df['weekly_return'].mean():.2%}, "
                        f"avg volume ratio={group_df['volume_ratio'].mean():.2f}"
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_stock_signal_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    把 True/False 信号转换成可读的信号列表。
    """

    rows = []

    for _, row in df.iterrows():
        ticker = row["ticker"]
        company = row.get("company", "")

        base_info = {
            "signal_date": datetime.now().strftime("%Y-%m-%d"),
            "ticker": ticker,
            "company": company,
            "daily_return": row.get("daily_return"),
            "weekly_return": row.get("weekly_return"),
            "monthly_return": row.get("monthly_return"),
            "volume_ratio": row.get("volume_ratio"),
            "group_type": "",
            "group_name": "",
        }

        if row.get("signal_big_up"):
            rows.append(
                {
                    **base_info,
                    "signal_type": "single_stock_big_up",
                    "message": f"{ticker} single-day gain >= 5%.",
                }
            )

        if row.get("signal_big_down"):
            rows.append(
                {
                    **base_info,
                    "signal_type": "single_stock_big_down",
                    "message": f"{ticker} single-day drop <= -5%.",
                }
            )

        if row.get("signal_volume_up"):
            rows.append(
                {
                    **base_info,
                    "signal_type": "volume_up",
                    "message": f"{ticker} rose with high volume.",
                }
            )

        if row.get("signal_volume_down"):
            rows.append(
                {
                    **base_info,
                    "signal_type": "volume_down",
                    "message": f"{ticker} fell with high volume.",
                }
            )

        if row.get("signal_near_52w_high"):
            rows.append(
                {
                    **base_info,
                    "signal_type": "near_52w_high",
                    "message": f"{ticker} is within 5% of its 52-week high.",
                }
            )

        if row.get("signal_strong_trend"):
            rows.append(
                {
                    **base_info,
                    "signal_type": "strong_trend",
                    "message": f"{ticker} is above 20/50/200-day moving averages.",
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    """
    主程序入口。
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Cannot find input file: {INPUT_FILE}")

    prices = pd.read_csv(INPUT_FILE)

    required_columns = [
        "ticker",
        "daily_return",
        "weekly_return",
        "monthly_return",
        "volume_ratio",
        "distance_to_52w_high",
        "above_20dma",
        "above_50dma",
        "above_200dma",
    ]

    for column in required_columns:
        if column not in prices.columns:
            raise ValueError(f"Missing required column in prices.csv: {column}")

    prices_with_signals = add_stock_signals(prices)

    stock_signal_rows = build_stock_signal_rows(prices_with_signals)
    theme_signal_rows = build_theme_signals(prices_with_signals)

    final_signals = pd.concat(
        [stock_signal_rows, theme_signal_rows],
        ignore_index=True,
    )

    final_signals.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved daily market signals to: {OUTPUT_FILE}\n")

    if final_signals.empty:
        print("No signals detected today.")
    else:
        print(final_signals[["signal_type", "ticker", "group_name", "message"]].head(30))


if __name__ == "__main__":
    main()
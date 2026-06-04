"""
update_prices.py

这个脚本的作用：
1. 读取 data/company_master.csv 里的股票列表
2. 使用 yfinance 下载每只股票最近一年的日线价格和成交量
3. 计算每日涨跌幅、周涨跌幅、月涨跌幅、均线、成交量异动等指标
4. 输出 data/prices.csv

这个文件是 Research OS 的“价格数据层”。
之后 daily_market_monitor.py 和 weekly_review.py 都会依赖它。
"""

from pathlib import Path
from datetime import datetime

import pandas as pd
import yfinance as yf

from history_utils import upsert_daily_history


# 项目根目录：
# 当前文件在 investment_os/src/update_prices.py
# parents[1] 表示向上两层，回到 investment_os
BASE_DIR = Path(__file__).resolve().parents[1]

# 输入文件：股票主表
INPUT_FILE = BASE_DIR / "data" / "company_master.csv"

# 输出文件：价格指标表
OUTPUT_FILE = BASE_DIR / "data" / "prices.csv"

# 历史价格指标表
HISTORY_FILE = (
    BASE_DIR
    / "data"
    / "prices_history.csv"
)


def calculate_price_features(ticker: str) -> dict:
    """
    下载一只股票最近一年的价格数据，并计算价格相关指标。

    参数：
        ticker: 股票代码，例如 "NVDA"

    返回：
        一个 dict，里面包含这只股票的价格指标。
    """

    # 下载最近 1 年的日线数据。
    # period="1y" 表示最近一年。
    # interval="1d" 表示日线。
    hist = yf.download(
        ticker,
        period="1y",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    # 如果没有下载到数据，就抛出错误。
    if hist.empty:
        raise ValueError(f"No price data returned for {ticker}")

    # 如果 yfinance 返回 MultiIndex columns，把它压平成普通 columns。
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

    close = hist["Close"]
    volume = hist["Volume"]

    latest_close = close.iloc[-1]

    # 日涨跌幅：
    # 最新收盘价 / 前一天收盘价 - 1
    daily_return = close.iloc[-1] / close.iloc[-2] - 1 if len(close) >= 2 else None

    # 周涨跌幅：
    # 最新收盘价 / 5个交易日前收盘价 - 1
    weekly_return = close.iloc[-1] / close.iloc[-6] - 1 if len(close) >= 6 else None

    # 月涨跌幅：
    # 最新收盘价 / 21个交易日前收盘价 - 1
    monthly_return = close.iloc[-1] / close.iloc[-22] - 1 if len(close) >= 22 else None

    # 均线：
    # 20日、50日、200日收盘价平均值
    ma20 = close.rolling(window=20).mean().iloc[-1] if len(close) >= 20 else None
    ma50 = close.rolling(window=50).mean().iloc[-1] if len(close) >= 50 else None
    ma200 = close.rolling(window=200).mean().iloc[-1] if len(close) >= 200 else None

    # 52周最高价和最低价
    high_52w = close.max()
    low_52w = close.min()

    # 距离52周高点：
    # 如果结果是 -0.10，表示当前价格比52周高点低10%
    distance_to_52w_high = latest_close / high_52w - 1 if high_52w else None

    # 成交量比率：
    # 最新成交量 / 20日平均成交量
    avg_volume_20d = volume.rolling(window=20).mean().iloc[-1] if len(volume) >= 20 else None
    latest_volume = volume.iloc[-1]

    volume_ratio = latest_volume / avg_volume_20d if avg_volume_20d is not None and avg_volume_20d != 0 else None

    return {
        "ticker": ticker,
        "latest_close": latest_close,
        "daily_return": daily_return,
        "weekly_return": weekly_return,
        "monthly_return": monthly_return,
        "ma20": ma20,
        "ma50": ma50,
        "ma200": ma200,
        "above_20dma": latest_close > ma20 if ma20 is not None else None,
        "above_50dma": latest_close > ma50 if ma50 is not None else None,
        "above_200dma": latest_close > ma200 if ma200 is not None else None,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "distance_to_52w_high": distance_to_52w_high,
        "latest_volume": latest_volume,
        "avg_volume_20d": avg_volume_20d,
        "volume_ratio": volume_ratio,
        "updated_at": datetime.now().strftime("%Y-%m-%d"),
    }


def main() -> None:
    """
    主程序入口。
    """

    # 检查 company_master.csv 是否存在。
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Cannot find input file: {INPUT_FILE}")

    # 读取股票主表。
    companies = pd.read_csv(INPUT_FILE)

    # 检查 ticker 列是否存在。
    if "ticker" not in companies.columns:
        raise ValueError("company_master.csv must contain a 'ticker' column")

    results = []

    # 遍历每只股票。
    for ticker in companies["ticker"]:
        ticker = str(ticker).strip().upper()

        try:
            print(f"Fetching price data for {ticker}...")
            row = calculate_price_features(ticker)
            results.append(row)

        except Exception as e:
            print(f"Error fetching price data for {ticker}: {e}")

    # 转成 DataFrame。
    price_df = pd.DataFrame(results)

    # 把 company_master.csv 里的产业链信息合并进来。
    final_df = companies.merge(
        price_df,
        on="ticker",
        how="left",
    )

    # 保存结果。
    final_df.to_csv(OUTPUT_FILE, index=False)

    upsert_daily_history(
        final_df,
        HISTORY_FILE,
    )

    print(f"\nSaved price data to: {OUTPUT_FILE}")
    print(final_df.head())


if __name__ == "__main__":
    main()

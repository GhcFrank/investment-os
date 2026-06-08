"""
update_fundamentals.py

这个脚本的作用：
1. 读取 data/master/company_master.csv
2. 从 yfinance 获取每只股票的基础财务和估值数据
3. 把手工维护的公司信息和自动抓取的数据合并
4. 输出 data/market_data/fundamentals.csv

company_master.csv 是你的“股票主表”。
fundamentals.csv 是程序生成的“基本面数据表”。
"""

from pathlib import Path
from datetime import datetime

import pandas as pd
import yfinance as yf

from utils.date_utils import today_et_str


# BASE_DIR 表示项目根目录，也就是 investment_os 这个文件夹。
# 假设当前文件路径是：
# investment_os/src/market_data/update_fundamentals.py
#
# Path(__file__) 是当前 Python 文件的路径。
# resolve() 把路径转换成完整绝对路径。
# parents[2] 表示向上走两层：
# update_fundamentals.py -> src -> investment_os
BASE_DIR = Path(__file__).resolve().parents[2]

# 输入文件：你手工维护的股票主表。
INPUT_FILE = BASE_DIR / "data" / "master" / "company_master.csv"

# 输出文件：程序自动生成的基本面数据表。
OUTPUT_FILE = BASE_DIR / "data" / "market_data" / "fundamentals.csv"


def get_company_data(ticker: str) -> dict:
    """
    从 yfinance 获取一只股票的基础数据。

    参数：
        ticker:
            股票代码，例如 "NVDA"、"CRDO"、"ANET"

    返回：
        一个 Python 字典 dict。
        字典里面保存这只股票的基本面数据。
    """

    # yf.Ticker(ticker) 会创建一个股票对象。
    # 这个对象可以用来访问该股票的行情、财务、估值等数据。
    stock = yf.Ticker(ticker)

    # stock.info 会返回一个字典，里面包含很多字段。
    # 注意：yfinance 的数据不是官方财报数据库，可能会有缺失或延迟。
    info = stock.info

    # info.get("字段名") 的意思是：
    # 尝试读取这个字段。
    # 如果字段不存在，就返回 None，而不是让程序报错。
    return {
        "ticker": ticker,

        # 市值
        "market_cap": info.get("marketCap"),

        # 最近十二个月收入
        "revenue": info.get("totalRevenue"),

        # 收入同比增长率
        "revenue_growth": info.get("revenueGrowth"),

        # 毛利率
        "gross_margin": info.get("grossMargins"),

        # 经营利润率
        "operating_margin": info.get("operatingMargins"),

        # 未来市盈率
        "forward_pe": info.get("forwardPE"),

        # 过去十二个月市盈率
        "trailing_pe": info.get("trailingPE"),

        # 企业价值 / 收入
        "ev_to_revenue": info.get("enterpriseToRevenue"),

        # 更新时间
        "updated_at": today_et_str(),
    }


def main() -> None:
    """
    主程序入口。

    当你运行：
        python src/market_data/update_fundamentals.py

    Python 会从这里开始执行。
    """

    # 检查 company_master.csv 是否存在。
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Cannot find input file: {INPUT_FILE}")

    # 读取 company_master.csv。
    # 这张表是你手工维护的公司主数据。
    companies = pd.read_csv(INPUT_FILE)

    # 确认里面有 ticker 这一列。
    # 没有 ticker，程序就不知道要抓哪些股票。
    if "ticker" not in companies.columns:
        raise ValueError("company_master.csv must contain a 'ticker' column")

    # 用来保存每只股票抓取结果的列表。
    results = []

    # 遍历 company_master.csv 里的每个 ticker。
    for ticker in companies["ticker"]:
        # 转成字符串，去掉空格，再转成大写。
        ticker = str(ticker).strip().upper()

        try:
            print(f"Fetching {ticker}...")

            # 抓取这只股票的数据。
            row = get_company_data(ticker)

            # 把结果加入 results 列表。
            results.append(row)

        except Exception as e:
            # 如果某只股票抓取失败，不让整个程序停止。
            # 打印错误，然后继续下一只股票。
            print(f"Error fetching {ticker}: {e}")

    # 把抓取结果转换成 DataFrame。
    fundamentals_df = pd.DataFrame(results)

    # 把 company_master.csv 的手工字段和 yfinance 的自动字段合并。
    #
    # on="ticker":
    #   按 ticker 这一列对齐。
    #
    # how="left":
    #   以 company_master.csv 为主。
    #   即使某只股票 yfinance 抓取失败，也保留这只股票。
    final_df = companies.merge(
        fundamentals_df,
        on="ticker",
        how="left"
    )

    # 保存到 data/market_data/fundamentals.csv。
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved fundamentals to: {OUTPUT_FILE}")

    # 打印前几行，方便确认结果。
    print(final_df.head())


if __name__ == "__main__":
    main()

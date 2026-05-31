"""
rank_stocks.py

这个脚本的作用：
1. 读取 data/fundamentals.csv
2. 根据增长、毛利率、经营利润率、估值等指标给股票打分
3. 输出 data/ranked_stocks.csv

注意：
这不是投资建议，只是一个研究排序工具。
它的目标是帮我们决定“先研究哪些公司”。
"""

from pathlib import Path

import pandas as pd


# 找到项目根目录。
# __file__ 表示当前这个 Python 文件的位置。
# parents[1] 表示往上走两层：
# src/rank_stocks.py -> src -> investment_os
BASE_DIR = Path(__file__).resolve().parents[1]

# 输入文件：基本面数据
INPUT_FILE = BASE_DIR / "data" / "fundamentals.csv"

# 输出文件：排序后的股票列表
OUTPUT_FILE = BASE_DIR / "data" / "ranked_stocks.csv"


def min_max_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """
    把一列数字转换成 0 到 100 的分数。

    举例：
    如果 revenue_growth 里面最高是 2.0，最低是 0.2，
    那么最高的公司得 100 分，最低的公司得 0 分。

    higher_is_better=True:
        数值越高越好，比如 revenue_growth、gross_margin。

    higher_is_better=False:
        数值越低越好，比如 forward_pe、ev_to_revenue。
    """

    # 把数据转换成数字。
    # 如果某个值无法转换成数字，会变成 NaN。
    numeric_series = pd.to_numeric(series, errors="coerce")

    min_value = numeric_series.min()
    max_value = numeric_series.max()

    # 如果这一列全是空值，或者所有值都一样，就返回 50 分。
    # 这样可以避免除以 0。
    if pd.isna(min_value) or pd.isna(max_value) or min_value == max_value:
        return pd.Series([50] * len(series), index=series.index)

    score = (numeric_series - min_value) / (max_value - min_value) * 100

    # 如果数值越低越好，就反过来打分。
    if not higher_is_better:
        score = 100 - score

    # 缺失值给 50 分，表示中性。
    return score.fillna(50)


def main() -> None:
    """
    主程序入口。
    """

    # 检查 fundamentals.csv 是否存在。
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Cannot find input file: {INPUT_FILE}")

    # 读取 fundamentals.csv。
    df = pd.read_csv(INPUT_FILE)

    # 检查必须存在的列。
    required_columns = [
        "ticker",
        "revenue_growth",
        "gross_margin",
        "operating_margin",
        "forward_pe",
        "ev_to_revenue",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    # 增长分数：收入增长越高越好。
    df["growth_score"] = min_max_score(df["revenue_growth"], higher_is_better=True)

    # 毛利率分数：毛利率越高越好。
    df["gross_margin_score"] = min_max_score(df["gross_margin"], higher_is_better=True)

    # 经营利润率分数：经营利润率越高越好。
    df["operating_margin_score"] = min_max_score(df["operating_margin"], higher_is_better=True)

    # Forward PE 分数：估值越低越好。
    df["forward_pe_score"] = min_max_score(df["forward_pe"], higher_is_better=False)

    # EV / Revenue 分数：估值越低越好。
    df["ev_to_revenue_score"] = min_max_score(df["ev_to_revenue"], higher_is_better=False)

    # 综合分数。
    # 第一版先用简单权重：
    # 增长 35%
    # 毛利率 20%
    # 经营利润率 20%
    # Forward PE 15%
    # EV / Revenue 10%
    df["total_score"] = (
        df["growth_score"] * 0.35
        + df["gross_margin_score"] * 0.20
        + df["operating_margin_score"] * 0.20
        + df["forward_pe_score"] * 0.15
        + df["ev_to_revenue_score"] * 0.10
    )

    # 按综合分数从高到低排序。
    df = df.sort_values(by="total_score", ascending=False)

    # 为了更容易阅读，把分数保留 2 位小数。
    score_columns = [
        "growth_score",
        "gross_margin_score",
        "operating_margin_score",
        "forward_pe_score",
        "ev_to_revenue_score",
        "total_score",
    ]

    df[score_columns] = df[score_columns].round(2)

    # 保存结果。
    df.to_csv(OUTPUT_FILE, index=False)

    # 在终端打印最重要的几列。
    print("\nStock ranking completed.\n")
    print(df[["ticker", "total_score", "revenue_growth", "gross_margin", "forward_pe", "ev_to_revenue"]])

    print(f"\nSaved ranked stocks to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
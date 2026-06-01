"""
build_sector_strength.py

作用：

1. 读取 data/prices.csv

2. 按 subtheme 聚合

例如：

GPU
Networking
Optical
Power
Energy

3. 计算主题强弱

4. 输出：

data/sector_strength.csv

5. 保存历史记录：

data/sector_strength_history.csv

未来：

daily_market_monitor.py
weekly_review.py
sector_rotation.py

都会依赖这些数据。
"""

from pathlib import Path
from datetime import datetime

import pandas as pd


# ============================================================
# 项目路径
# ============================================================

# 当前文件：
# investment_os/src/build_sector_strength.py
#
# parents[1]
# ->
# investment_os
BASE_DIR = Path(__file__).resolve().parents[1]

# 输入文件
INPUT_FILE = BASE_DIR / "data" / "prices.csv"

# 当天最新结果
OUTPUT_FILE = BASE_DIR / "data" / "sector_strength.csv"

# 历史数据库
HISTORY_FILE = (
    BASE_DIR
    / "data"
    / "sector_strength_history.csv"
)


# ============================================================
# 计算主题强度
# ============================================================

def calculate_sector_scores(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    根据 prices.csv 计算主题强度。

    输入：

        prices.csv

    输出：

        subtheme
        daily_return
        weekly_return
        monthly_return
        volume_ratio

        daily_score
        weekly_score
        monthly_score
        volume_score

        sector_strength_score
    """

    grouped = (
        df.groupby("subtheme")
        .agg(
            {
                "daily_return": "mean",
                "weekly_return": "mean",
                "monthly_return": "mean",
                "volume_ratio": "mean",
            }
        )
        .reset_index()
    )

    # ========================================================
    # 百分位排名
    #
    # rank(pct=True)
    #
    # 返回：
    #
    # 0 ~ 1
    #
    # 再乘100
    #
    # 变成：
    #
    # 0 ~ 100
    # ========================================================

    grouped["daily_score"] = (
        grouped["daily_return"]
        .rank(pct=True)
        * 100
    )

    grouped["weekly_score"] = (
        grouped["weekly_return"]
        .rank(pct=True)
        * 100
    )

    grouped["monthly_score"] = (
        grouped["monthly_return"]
        .rank(pct=True)
        * 100
    )

    grouped["volume_score"] = (
        grouped["volume_ratio"]
        .rank(pct=True)
        * 100
    )

    # ========================================================
    # 第一版主题强度公式
    # ========================================================

    grouped["sector_strength_score"] = (
        grouped["daily_score"] * 0.20
        + grouped["weekly_score"] * 0.40
        + grouped["monthly_score"] * 0.30
        + grouped["volume_score"] * 0.10
    )

    grouped["sector_strength_score"] = (
        grouped["sector_strength_score"]
        .round(2)
    )

    grouped["updated_at"] = (
        datetime.now()
        .strftime("%Y-%m-%d")
    )

    # ========================================================
    # 按强度排序
    # ========================================================

    grouped = grouped.sort_values(
        by="sector_strength_score",
        ascending=False,
    )

    # 重新生成索引
    grouped = grouped.reset_index(
        drop=True
    )

    return grouped


# ============================================================
# 保存历史记录
# ============================================================

def append_history(
    sector_strength: pd.DataFrame,
) -> None:
    """
    把当天结果写入历史数据库。

    第一版逻辑：

        文件不存在
            -> 创建

        文件存在
            -> 追加

    注意：

        如果一天运行多次，
        会产生重复记录。

    下一版再解决。
    """

    history_df = sector_strength.copy()

    history_df["date"] = (
        datetime.now()
        .strftime("%Y-%m-%d")
    )

    # 第一次运行
    if not HISTORY_FILE.exists():

        history_df.to_csv(
            HISTORY_FILE,
            index=False,
        )

        print(
            f"\nCreated history file:\n"
            f"{HISTORY_FILE}\n"
        )

        return

    # 读取已有历史
    existing = pd.read_csv(
        HISTORY_FILE
    )

    combined = pd.concat(
        [
            existing,
            history_df,
        ],
        ignore_index=True,
    )

    combined.to_csv(
        HISTORY_FILE,
        index=False,
    )

    print(
        f"\nUpdated history file:\n"
        f"{HISTORY_FILE}\n"
    )


# ============================================================
# 主程序
# ============================================================

def main() -> None:
    """
    主程序入口。
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE}"
        )

    prices = pd.read_csv(
        INPUT_FILE
    )

    if "subtheme" not in prices.columns:
        raise ValueError(
            "prices.csv missing subtheme column"
        )

    sector_strength = (
        calculate_sector_scores(
            prices
        )
    )

    # 保存当天结果
    sector_strength.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # 保存历史
    append_history(
        sector_strength
    )

    print(
        f"\nSaved sector strength to:\n"
        f"{OUTPUT_FILE}\n"
    )

    print(
        sector_strength[
            [
                "subtheme",
                "sector_strength_score",
                "daily_return",
                "weekly_return",
                "monthly_return",
            ]
        ]
    )


if __name__ == "__main__":
    main()
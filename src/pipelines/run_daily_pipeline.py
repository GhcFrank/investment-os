"""
run_daily_pipeline.py

作用：
按顺序运行每日收盘后的 Research OS 数据流水线。

当前每日流程：

1. 更新股票价格和成交量数据
   -> src/market_data/update_prices.py
   -> 输出 data/market_data/prices.csv

2. 更新市场情绪指标
   -> src/market_data/update_sentiment_indicators.py
   -> 输出 data/market_data/vix.csv
   -> 输出 data/market_data/cnn_fear_greed.csv

3. 计算板块强度
   -> src/signals/build_sector_strength.py
   -> 输出 data/signals/sector_strength.csv
   -> 更新 data/signals/sector_strength_history.csv

4. 生成每日市场信号
   -> src/signals/daily_market_monitor.py
   -> 输出 data/signals/daily_market_signals.csv

5. 检查明天是否有财报
   -> src/events/check_earnings_calendar.py
   -> 如果命中，发送邮件提醒
   -> 更新 data/events/earnings_alert_history.csv

6. 检查 SEC EDGAR 重要 filing
   -> src/events/check_sec_filings.py
   -> 如果发现新 filing，发送邮件提醒
   -> 更新 data/events/sec_filings.csv

7. 更新 Polymarket earnings 预测数据
   -> src/prediction_markets/update_polymarket_earnings_markets.py
   -> src/prediction_markets/match_polymarket_earnings.py
   -> src/prediction_markets/update_polymarket_predictions.py
   -> src/prediction_markets/check_polymarket_prediction_signals.py

为什么要有这个文件？

以前 GitHub Actions 需要分别运行多个脚本。
有了这个总控文件后，GitHub Actions 只需要运行：

    python src/pipelines/run_daily_pipeline.py

这样项目结构更清晰，后续加日报邮件、异常检查、日志记录也更方便。
"""

from pathlib import Path
import os
import subprocess
import sys
from datetime import datetime


# 项目根目录：
# 当前文件在 investment_os/src/pipelines/run_daily_pipeline.py
# parents[2] 表示向上三层回到 investment_os
BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.send_email import send_email
from market_data.sentiment_summary import build_sentiment_email_section


def run_script(script_path: Path) -> None:
    """
    运行一个 Python 脚本。

    参数：
        script_path:
            要运行的脚本路径，例如：
            investment_os/src/market_data/update_prices.py

    如果脚本运行失败：
        直接抛出错误，让整个 pipeline 停止。
    """

    print("=" * 80)
    print(f"Running: {script_path}")
    print("=" * 80)

    # sys.executable 表示当前正在运行的 Python 解释器。
    #
    # 在本地虚拟环境中，它会是：
    # /home/gooder/investment_os/myInvestmentEnv/bin/python
    #
    # 在 GitHub Actions 中，它会是 GitHub 设置好的 Python。
    #
    # 这样可以保证子脚本使用同一个 Python 环境。
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR,
        env={
            **os.environ,
            "PYTHONPATH": str(SRC_DIR),
        },
        text=True,
    )

    # returncode 等于 0 表示脚本成功。
    # 非 0 表示脚本失败。
    if result.returncode != 0:
        raise RuntimeError(
            f"Script failed: {script_path}"
        )


def main() -> None:
    """
    每日 pipeline 主入口。
    """

    print("\nDaily pipeline started.")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {BASE_DIR}\n")

    scripts = [
        BASE_DIR / "src" / "market_data" / "update_prices.py",
        BASE_DIR / "src" / "market_data" / "update_sentiment_indicators.py",
        BASE_DIR / "src" / "signals" / "build_sector_strength.py",
        BASE_DIR / "src" / "signals" / "daily_market_monitor.py",
        BASE_DIR / "src" / "events" / "check_earnings_calendar.py",
        BASE_DIR / "src" / "events" / "check_sec_filings.py",
        BASE_DIR / "src" / "prediction_markets" / "update_polymarket_earnings_markets.py",
        BASE_DIR / "src" / "prediction_markets" / "match_polymarket_earnings.py",
        BASE_DIR / "src" / "prediction_markets" / "update_polymarket_predictions.py",
        BASE_DIR / "src" / "prediction_markets" / "check_polymarket_prediction_signals.py",
    ]

    for script in scripts:
        if not script.exists():
            raise FileNotFoundError(
                f"Cannot find script: {script}"
            )

        run_script(script)

    print("\nDaily pipeline completed successfully.")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    sentiment_section = build_sentiment_email_section()

    # 发送完成通知邮件。
    # 这里复用 src/utils/send_email.py 里的 send_email 函数。
    send_email(
        subject="Investment OS Daily Pipeline Completed",
        body=(
            "Daily pipeline completed successfully.\n\n"
            f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"{sentiment_section}\n\n"
            "Generated files:\n"
            "- data/market_data/prices.csv\n"
            "- data/market_data/prices_history.csv\n"
            "- data/market_data/vix.csv\n"
            "- data/market_data/vix_history.csv\n"
            "- data/market_data/cnn_fear_greed.csv\n"
            "- data/market_data/cnn_fear_greed_history.csv\n"
            "- data/market_data/sentiment_fetch_log.csv\n"
            "- data/signals/sector_strength.csv\n"
            "- data/signals/sector_strength_history.csv\n"
            "- data/signals/daily_market_signals.csv\n"
            "- data/events/earnings_alert_history.csv (when an alert is sent)\n"
            "- data/events/sec_company_map.csv\n"
            "- data/events/sec_filings.csv\n"
            "- data/events/sec_alert_history.csv (when an alert is sent)\n"
            "- data/events/sec_initialized.flag\n"
            "- data/prediction_markets/polymarket_earnings_markets.csv\n"
            "- data/prediction_markets/polymarket_earnings_watchlist.csv\n"
            "- data/prediction_markets/polymarket_predictions.csv\n"
            "- data/prediction_markets/polymarket_predictions_history.csv\n"
        ),
    )


if __name__ == "__main__":
    main()

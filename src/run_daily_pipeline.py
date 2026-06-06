"""
run_daily_pipeline.py

作用：
按顺序运行每日收盘后的 Research OS 数据流水线。

当前每日流程：

1. 更新股票价格和成交量数据
   -> src/update_prices.py
   -> 输出 data/prices.csv

2. 计算板块强度
   -> src/build_sector_strength.py
   -> 输出 data/sector_strength.csv
   -> 更新 data/sector_strength_history.csv

3. 生成每日市场信号
   -> src/daily_market_monitor.py
   -> 输出 data/daily_market_signals.csv

4. 检查明天是否有财报
   -> src/check_earnings_calendar.py
   -> 如果命中，发送邮件提醒
   -> 更新 data/earnings_alert_history.csv

为什么要有这个文件？

以前 GitHub Actions 需要分别运行多个脚本。
有了这个总控文件后，GitHub Actions 只需要运行：

    python src/run_daily_pipeline.py

这样项目结构更清晰，后续加日报邮件、异常检查、日志记录也更方便。
"""

from pathlib import Path
import subprocess
import sys
from datetime import datetime
from send_email import send_email


# 项目根目录：
# 当前文件在 investment_os/src/run_daily_pipeline.py
# parents[1] 表示向上两层回到 investment_os
BASE_DIR = Path(__file__).resolve().parents[1]


def run_script(script_path: Path) -> None:
    """
    运行一个 Python 脚本。

    参数：
        script_path:
            要运行的脚本路径，例如：
            investment_os/src/update_prices.py

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
        BASE_DIR / "src" / "update_prices.py",
        BASE_DIR / "src" / "build_sector_strength.py",
        BASE_DIR / "src" / "daily_market_monitor.py",
        BASE_DIR / "src" / "check_earnings_calendar.py",
    ]

    for script in scripts:
        if not script.exists():
            raise FileNotFoundError(
                f"Cannot find script: {script}"
            )

        run_script(script)

    print("\nDaily pipeline completed successfully.")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # 发送完成通知邮件。
    # 这里复用 src/send_email.py 里的 send_email 函数。
    send_email(
        subject="Investment OS Daily Pipeline Completed",
        body=(
            "Daily pipeline completed successfully.\n\n"
            f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            "Generated files:\n"
            "- data/prices.csv\n"
            "- data/prices_history.csv\n"
            "- data/sector_strength.csv\n"
            "- data/sector_strength_history.csv\n"
            "- data/daily_market_signals.csv\n"
            "- data/earnings_alert_history.csv (when an alert is sent)\n"
        ),
    )


if __name__ == "__main__":
    main()

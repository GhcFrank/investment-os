"""
run_daily_pipeline.py

作用：
按顺序运行每日收盘后的 Research OS 数据流水线。

当前每日流程：

1. 更新 VIX 市场数据（daily 正式路径不运行 CNN）
   -> src/market_data/update_sentiment_indicators.py
   -> 输出 data/market_data/vix.csv 和 vix_history.csv

2. 构建正式 VIX market sentiment signal
   -> src/signals/build_vix_market_sentiment.py
   -> 输出 data/signals/vix_market_sentiment.csv

3. 更新 Vast.ai GPU 云市场原始报价历史（仅 Search Offers，只读）
   -> src/market_data/update_gpu_cloud_market.py
   -> 输出 data/market_data/gpu_cloud_market_history.csv
   -> 输出 data/market_data/gpu_cloud_market_fetch_log.csv

4. 构建 Vast.ai GPU 云市场价格、库存和可用性信号
   -> src/signals/build_gpu_cloud_market_signals.py
   -> 输出 data/signals/gpu_cloud_market_signals.csv

5. 更新股票价格和成交量数据
   -> src/market_data/update_prices.py
   -> 输出 data/market_data/prices.csv

6. 更新 11 只 GICS 一级板块 ETF 的 State Street 官方基金历史
   -> src/market_data/update_sector_etf_fund_history.py
   -> 输出 data/market_data/sector_etf_fund_history/<configured-filename>.csv

7. 更新 SOXX、IGV 的 iShares 官方基金历史
   -> src/market_data/update_ishares_etf_fund_history.py
   -> 输出 data/market_data/sector_etf_fund_history/<configured-filename>.csv

8. 更新 13 只 leadership ETF 市场价格
   -> src/market_data/update_sector_etf_prices.py
   -> 输出 data/market_data/sector_etf_prices.csv

9. 构建 13 只 leadership ETF 30/90/250 交易日复权价格涨幅
   -> src/signals/build_sector_etf_metrics.py
   -> 输出 data/signals/sector_etf_metrics/<configured-filename>.csv

10. 构建 13 只 ETF leadership 每日排名，内容并入统一 daily email
   -> src/signals/build_sector_etf_rankings.py
   -> 输出 data/signals/sector_etf_daily_rankings.csv

11. 计算板块强度
   -> src/signals/build_sector_strength.py
   -> 输出 data/signals/sector_strength.csv
   -> 更新 data/signals/sector_strength_history.csv

12. 生成每日市场信号
   -> src/signals/daily_market_monitor.py
   -> 输出 data/signals/daily_market_signals.csv

13. 检查明天是否有财报
   -> src/events/check_earnings_calendar.py
   -> 如果命中，发送邮件提醒
   -> 更新 data/events/earnings_alert_history.csv

14. 检查 SEC EDGAR 重要 filing
   -> src/events/check_sec_filings.py
   -> 如果发现新 filing，发送邮件提醒
   -> 更新 data/events/sec_filings.csv

15. 更新 Polymarket earnings 预测数据
   -> src/prediction_markets/update_polymarket_earnings_markets.py
   -> src/prediction_markets/match_polymarket_earnings.py
   -> src/prediction_markets/update_polymarket_predictions.py
   -> src/prediction_markets/check_polymarket_prediction_signals.py

为什么要有这个文件？

以前 GitHub Actions 需要分别运行多个脚本。
有了这个总控文件后，GitHub Actions 只需要运行：

    PYTHONPATH=src python -m pipelines.run_daily_pipeline

这样项目结构更清晰，后续加日报邮件、异常检查、日志记录也更方便。
"""

from dataclasses import dataclass
from html import escape
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
from market_data.gpu_cloud_summary import (
    GPUCloudEmailSection,
    build_gpu_cloud_email_section,
    build_gpu_cloud_unavailable_email_section,
)
from market_data.gpu_cloud_status import (
    API_KEY_MISSING,
    PROVIDER_ERROR,
    SCHEMA_ERROR,
)
from market_data.update_sentiment_indicators import run_vix_market_update
from market_data.vix_sentiment_summary import (
    VIXMarketSentimentEmailSection,
    build_vix_market_sentiment_email_section,
    build_vix_market_sentiment_unavailable_email_section,
)
from market_data.update_sector_etf_fund_history import (
    run_sector_etf_fund_history_update,
)
from market_data.update_ishares_etf_fund_history import (
    run_ishares_etf_fund_history_update,
)
from market_data.update_sector_etf_prices import run_sector_etf_price_update
from market_data.update_gpu_cloud_market import run_gpu_cloud_market_update
from market_data.vast_ai_client import (
    VastAIAuthenticationError,
    VastAISchemaError,
)
from signals.build_gpu_cloud_market_signals import (
    run_gpu_cloud_market_signals_update,
)
from signals.build_sector_etf_metrics import run_sector_etf_metrics_update
from signals.build_sector_etf_rankings import (
    run_sector_etf_daily_ranking,
)
from signals.build_vix_market_sentiment import (
    run_vix_market_sentiment_signals_update,
)


@dataclass(frozen=True)
class GPUCloudPipelineResult:
    available: bool
    status: str


@dataclass(frozen=True)
class VIXPipelineResult:
    available: bool
    status: str


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


def run_sector_etf_price_step():
    """
    Update Yahoo sector ETF market prices in-process.
    """

    print("=" * 80)
    print("Running: Yahoo ETF price update: 13 leadership ETFs")
    print("=" * 80)
    summary = run_sector_etf_price_update()
    print(summary.format())
    print("Completed: Yahoo ETF price update: 13 leadership ETFs")
    return summary


def run_gpu_cloud_market_step():
    """Collect a read-only Vast.ai Search Offers snapshot in-process."""

    print("=" * 80)
    print("Running: Vast.ai GPU cloud market update (Search Offers only)")
    print("=" * 80)
    summary = run_gpu_cloud_market_update()
    print(summary.format())
    print("Completed: Vast.ai GPU cloud market update")
    return summary


def run_vix_market_step():
    """Update only the VIX market observation; CNN remains manual-only."""

    print("=" * 80)
    print("Running: VIX market data update (CNN daily integration disabled)")
    print("=" * 80)
    summary = run_vix_market_update()
    print(summary.format())
    return summary


def run_vix_market_signals_step():
    """Build the formal VIX sentiment signal before email rendering."""

    print("=" * 80)
    print("Running: VIX market sentiment signal build")
    print("=" * 80)
    summary = run_vix_market_sentiment_signals_update()
    print(summary.format())
    return summary


def run_vix_steps_best_effort() -> VIXPipelineResult:
    """Keep the daily pipeline alive when VIX is unavailable."""

    print("[VIX] Updating market data...")
    try:
        update_summary = run_vix_market_step()
    except Exception:
        print("[VIX] WARNING: data unavailable; continuing daily pipeline.")
        return VIXPipelineResult(False, "DATA_UNAVAILABLE")
    if not update_summary.available:
        print("[VIX] WARNING: data unavailable; continuing daily pipeline.")
        return VIXPipelineResult(False, "DATA_UNAVAILABLE")
    if update_summary.warnings:
        print(
            "[VIX] WARNING: update completed with "
            f"{len(update_summary.warnings)} warning(s)."
        )

    print("[VIX] Building market sentiment signal...")
    try:
        signal_summary = run_vix_market_signals_step()
    except Exception:
        print("[VIX] WARNING: signal unavailable; continuing daily pipeline.")
        return VIXPipelineResult(False, "DATA_UNAVAILABLE")
    print("[VIX] Market sentiment update completed.")
    return VIXPipelineResult(True, signal_summary.data_status)


def run_gpu_cloud_market_signals_step():
    """Build the durable single-provider GPU cloud market signals."""

    print("=" * 80)
    print("Running: GPU cloud market price, inventory, and availability signals")
    print("Availability scope: Vast.ai only; not yet cross-provider")
    print("=" * 80)
    summary = run_gpu_cloud_market_signals_update()
    print(summary.format())
    print("Completed: GPU cloud market signals")
    return summary


def _safe_gpu_failure_status(error: Exception) -> str:
    if isinstance(error, VastAIAuthenticationError):
        return API_KEY_MISSING
    if isinstance(error, (VastAISchemaError, ValueError)):
        return SCHEMA_ERROR
    return PROVIDER_ERROR


def run_gpu_cloud_steps_best_effort() -> GPUCloudPipelineResult:
    """Run collection then signals without blocking the rest of the day."""

    print("[GPU CLOUD] Updating Vast.ai marketplace snapshot...")
    try:
        market_summary = run_gpu_cloud_market_step()
    except Exception as error:
        status = _safe_gpu_failure_status(error)
        print(
            f"[GPU CLOUD] WARNING: update unavailable ({status}); "
            "continuing daily pipeline."
        )
        return GPUCloudPipelineResult(available=False, status=status)

    warnings = getattr(market_summary, "warnings", ())
    warnings = warnings if isinstance(warnings, (tuple, list)) else ()
    pricing_types_failed = getattr(
        market_summary, "pricing_types_failed", 0
    )
    pricing_types_failed = (
        pricing_types_failed
        if isinstance(pricing_types_failed, int)
        else 0
    )
    if warnings or pricing_types_failed:
        print(
            "[GPU CLOUD] WARNING: snapshot completed with "
            f"{len(warnings)} warning(s) and "
            f"{pricing_types_failed} failed pricing type(s)."
        )

    print("[GPU CLOUD] Building daily GPU supply signals...")
    try:
        run_gpu_cloud_market_signals_step()
    except Exception:
        print(
            "[GPU CLOUD] WARNING: signal data unavailable "
            f"({SCHEMA_ERROR}); continuing daily pipeline."
        )
        return GPUCloudPipelineResult(
            available=False,
            status=SCHEMA_ERROR,
        )

    print("[GPU CLOUD] GPU market update completed.")
    return GPUCloudPipelineResult(available=True, status="SUCCESS")


def run_sector_etf_fund_history_step():
    """
    Update State Street NAV, shares, and total-net-assets history in-process.
    """

    print("=" * 80)
    print("Running: State Street ETF fund update: 11 primary-sector ETFs")
    print("=" * 80)
    summary = run_sector_etf_fund_history_update()
    print(summary.format())
    print("Completed: State Street ETF fund update: 11 primary-sector ETFs")
    return summary


def run_ishares_etf_fund_history_step():
    """Update official SOXX and IGV NAV, shares, and AUM history."""

    print("=" * 80)
    print("Running: iShares ETF fund update: 2 industry ETFs")
    print("=" * 80)
    summary = run_ishares_etf_fund_history_update()
    print(summary.format())
    if summary.failed:
        print(
            "WARNING: iShares fund data was incomplete; Yahoo price, "
            "metrics, and leadership ranking will continue because they use "
            "Yahoo adjusted close."
        )
    print("Completed: iShares ETF fund update: 2 industry ETFs")
    return summary


def run_sector_etf_metrics_step():
    """
    Rebuild local trading-day adjusted-close returns in-process.
    """

    print("=" * 80)
    print("Running: ETF metrics update: 13 leadership ETFs")
    print("ETF return horizons: 30/90/250 trading days")
    print("=" * 80)
    summary = run_sector_etf_metrics_update()
    print(summary.format())
    if (
        summary.failed
        or summary.succeeded != summary.configured_etfs
    ):
        raise RuntimeError(
            "Sector ETF metrics were incomplete; rankings and email are "
            "blocked"
        )
    print("Completed: ETF metrics update: 13 leadership ETFs")
    return summary


def run_sector_etf_rankings_step():
    """
    Build the latest same-date rankings for the unified daily email.
    """

    print("=" * 80)
    print("Running: Leadership ranking universe: 13 ETFs")
    print("=" * 80)
    summary = run_sector_etf_daily_ranking(send_email_message=False)
    print(summary.format())
    print("Completed: Leadership ranking universe: 13 ETFs")
    return summary


def _html_body_fragment(document: str) -> str:
    """Extract an existing email's body so it can be safely consolidated."""

    lower = document.lower()
    start = lower.find("<body")
    if start == -1:
        return document
    start = document.find(">", start)
    end = lower.rfind("</body>")
    if start == -1 or end == -1 or end <= start:
        return document
    return document[start + 1 : end]


def build_daily_pipeline_email(
    *,
    run_time: str,
    vix_section: VIXMarketSentimentEmailSection,
    gpu_section: GPUCloudEmailSection,
    ranking_email: object | None = None,
) -> tuple[str, str]:
    """Build the one plain/HTML daily email after every data step."""

    plain_sections = [
        "Daily pipeline completed successfully.",
        f"Run time: {run_time}",
    ]
    html_sections = [
        "<h1>Daily pipeline completed successfully</h1>",
        f"<p>Run time: {escape(run_time)}</p>",
    ]
    plain_sections.append(vix_section.plain_text)
    html_sections.append(vix_section.html)
    if ranking_email is not None:
        ranking_plain = str(getattr(ranking_email, "plain_text", "")).strip()
        ranking_html = str(getattr(ranking_email, "html", "")).strip()
        if ranking_plain:
            plain_sections.append(ranking_plain)
        if ranking_html:
            html_sections.append(_html_body_fragment(ranking_html))
    plain_sections.append(gpu_section.plain_text)
    html_sections.append(gpu_section.html)

    plain_body = "\n\n".join(plain_sections)
    html_body = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<style>body{font-family:Arial,sans-serif;color:#222}"
        "table{border-collapse:collapse;margin-bottom:18px}"
        "th,td{border:1px solid #ccc;padding:6px 8px;text-align:left}"
        "th{background:#f3f4f6}</style></head><body>"
        + "".join(html_sections)
        + "</body></html>"
    )
    return plain_body, html_body


def main() -> None:
    """
    每日 pipeline 主入口。
    """

    print("\nDaily pipeline started.")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {BASE_DIR}\n")

    # VIX-only data and its formal signal run before all other daily sections.
    # CNN remains available only through its manual module entrypoint.
    vix_result = run_vix_steps_best_effort()

    # This provider integration is deliberately read-only. It only searches
    # public offers and never creates, rents, starts, bids on, or destroys an
    # instance. Signals immediately follow collection so request failures are
    # not confused with zero inventory. GPU failures degrade only its email
    # section and never block the remaining daily market steps.
    gpu_result = run_gpu_cloud_steps_best_effort()

    ordinary_prices_script = (
        BASE_DIR / "src" / "market_data" / "update_prices.py"
    )
    if not ordinary_prices_script.exists():
        raise FileNotFoundError(
            f"Cannot find script: {ordinary_prices_script}"
        )
    run_script(ordinary_prices_script)

    # Sector ETF collection is deliberately separate from the existing AI
    # theme/subtheme strength data and runs before downstream signal steps.
    run_sector_etf_fund_history_step()
    run_ishares_etf_fund_history_step()
    run_sector_etf_price_step()
    run_sector_etf_metrics_step()
    ranking_summary = run_sector_etf_rankings_step()

    remaining_scripts = [
        BASE_DIR / "src" / "signals" / "build_sector_strength.py",
        BASE_DIR / "src" / "signals" / "daily_market_monitor.py",
        BASE_DIR / "src" / "events" / "check_earnings_calendar.py",
        BASE_DIR / "src" / "events" / "check_sec_filings.py",
        BASE_DIR / "src" / "prediction_markets" / "update_polymarket_earnings_markets.py",
        BASE_DIR / "src" / "prediction_markets" / "match_polymarket_earnings.py",
        BASE_DIR / "src" / "prediction_markets" / "update_polymarket_predictions.py",
        BASE_DIR / "src" / "prediction_markets" / "check_polymarket_prediction_signals.py",
    ]

    for script in remaining_scripts:
        if not script.exists():
            raise FileNotFoundError(
                f"Cannot find script: {script}"
            )

        run_script(script)

    print("\nDaily pipeline completed successfully.")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if vix_result.available:
        try:
            vix_section = build_vix_market_sentiment_email_section()
        except Exception:
            vix_section = (
                build_vix_market_sentiment_unavailable_email_section()
            )
    else:
        vix_section = build_vix_market_sentiment_unavailable_email_section(
            vix_result.status
        )
    if gpu_result.available:
        try:
            gpu_section = build_gpu_cloud_email_section()
        except Exception:
            gpu_section = build_gpu_cloud_unavailable_email_section(
                SCHEMA_ERROR
            )
    else:
        gpu_section = build_gpu_cloud_unavailable_email_section(
            gpu_result.status
        )
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plain_body, html_body = build_daily_pipeline_email(
        run_time=run_time,
        vix_section=vix_section,
        gpu_section=gpu_section,
        ranking_email=ranking_summary.email,
    )

    # 发送完成通知邮件。
    # 这里复用 src/utils/send_email.py 里的 send_email 函数。
    send_email(
        subject="Investment OS Daily Pipeline Completed",
        body=plain_body,
        html_body=html_body,
    )


if __name__ == "__main__":
    main()

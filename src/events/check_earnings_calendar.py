"""
check_earnings_calendar.py

作用：
1. 读取 data/master/company_master.csv
2. 使用 yfinance 查询每只股票的下一次 earnings date
3. 如果某家公司明天发布财报，发送邮件提醒
4. 保存当前查到的 earnings calendar 快照
5. 记录已提醒事项，避免重复发送

注意：
yfinance 的 earnings calendar 数据可能缺失、延迟或变化。
这个脚本适合做提醒，但不能当作权威财报日历。
"""

from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from utils.send_email import send_email


# ============================================================
# 路径设置
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "master" / "company_master.csv"

# 保存每次查到的 earnings calendar 快照。
# 这个文件方便你回看：系统当时查到了哪些财报日期。
CALENDAR_FILE = BASE_DIR / "data" / "events" / "earnings_calendar.csv"

# 保存已经发过提醒的记录。
# 这个文件用于避免同一家公司、同一个财报日期重复提醒。
HISTORY_FILE = BASE_DIR / "data" / "events" / "earnings_alert_history.csv"

MARKET_TIMEZONE = ZoneInfo("America/New_York")


# ============================================================
# 日期工具
# ============================================================

def today_et() -> date:
    """
    返回美东日期。

    GitHub Actions 默认使用 UTC 运行。
    但美股财报提醒应该按美东日期判断。
    """

    return datetime.now(MARKET_TIMEZONE).date()


def normalize_date(value) -> date | None:
    """
    把 yfinance 可能返回的日期类型统一成 Python date。

    yfinance 可能返回：
    - pandas Timestamp
    - datetime
    - date
    - 字符串
    - None
    - NaN

    返回：
    - date
    - None
    """

    # 先单独处理 None。
    # 不要直接写：if value is None or pd.isna(value)
    # 因为如果 value 是 list/array，pd.isna(value) 会返回数组，
    # 容易触发 “truth value is ambiguous” 错误。
    if value is None:
        return None

    # 如果是 pandas Timestamp
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.date()

    # 如果是 datetime
    if isinstance(value, datetime):
        return value.date()

    # 如果本来就是 date
    if isinstance(value, date):
        return value

    # 其他类型，例如字符串，尝试交给 pandas 解析。
    try:
        parsed = pd.to_datetime(value)

        if pd.isna(parsed):
            return None

        return parsed.date()

    except Exception:
        return None


# ============================================================
# yfinance 数据读取
# ============================================================

def get_earnings_dates(ticker: str) -> list[date]:
    """
    从 yfinance 获取一只股票的 earnings date。

    第一优先级：
        stock.get_calendar()

    第二优先级：
        stock.get_earnings_dates(limit=12)

    返回：
        排序后的 date 列表。
    """

    stock = yf.Ticker(ticker)
    dates: list[date] = []
    today = today_et()

    # --------------------------------------------------------
    # 方法1：get_calendar()
    # --------------------------------------------------------

    try:
        calendar = stock.get_calendar()

        earnings_dates = None

        # yfinance 一般返回 dict。
        if isinstance(calendar, dict):
            earnings_dates = (
                calendar.get("Earnings Date")
                or calendar.get("EarningsDate")
                or calendar.get("earningsDate")
            )

        # 有时可能返回 DataFrame。
        elif isinstance(calendar, pd.DataFrame):
            if "Earnings Date" in calendar.index:
                earnings_dates = calendar.loc["Earnings Date"].tolist()

        if earnings_dates is not None:
            # yfinance 有时返回单个 Timestamp，
            # 有时返回 list/tuple。
            if not isinstance(earnings_dates, (list, tuple, pd.Series)):
                earnings_dates = [earnings_dates]

            for earnings_date in earnings_dates:
                normalized = normalize_date(earnings_date)

                if normalized is not None:
                    dates.append(normalized)

    except Exception as error:
        print(f"Warning: calendar lookup failed for {ticker}: {error}")

    if dates:
        future_dates = [earnings_date for earnings_date in sorted(set(dates)) if earnings_date >= today]
        return future_dates

    # --------------------------------------------------------
    # 方法2：get_earnings_dates()
    # --------------------------------------------------------

    try:
        earnings_df = stock.get_earnings_dates(limit=12)

        if earnings_df is None or earnings_df.empty:
            return []

        candidate_values = list(earnings_df.index)

        if "Earnings Date" in earnings_df.columns:
            candidate_values.extend(
                earnings_df["Earnings Date"].tolist()
            )

        for earnings_date in candidate_values:
            normalized = normalize_date(earnings_date)

            if normalized is not None:
                dates.append(normalized)

    except Exception as error:
        print(f"Warning: earnings dates lookup failed for {ticker}: {error}")

    future_dates = [earnings_date for earnings_date in sorted(set(dates)) if earnings_date >= today]
    return future_dates


# ============================================================
# 文件读取与保存
# ============================================================

def load_alert_history() -> pd.DataFrame:
    """
    读取已发送提醒记录。

    如果文件不存在，或者文件存在但内容为空，
    就返回一个带标准列名的空 DataFrame。
    """

    columns = [
        "alert_date",
        "ticker",
        "company",
        "earnings_date",
        "sent_at",
    ]

    if not HISTORY_FILE.exists():
        return pd.DataFrame(columns=columns)

    try:
        return pd.read_csv(HISTORY_FILE)

    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)


def already_alerted(
    history: pd.DataFrame,
    ticker: str,
    earnings_date: date,
) -> bool:
    """
    判断同一 ticker、同一 earnings_date 是否已经提醒过。
    """

    if history.empty:
        return False

    matched = history[
        (history["ticker"] == ticker)
        & (history["earnings_date"] == earnings_date.isoformat())
    ]

    return not matched.empty


def save_calendar_snapshot(
    calendar_rows: list[dict],
) -> None:
    """
    保存当前查到的 earnings calendar 快照。

    输出：
        data/events/earnings_calendar.csv

    这个文件不是提醒历史。
    它只是记录当前系统看到的每家公司下一次 earnings date。
    """

    CALENDAR_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    calendar_df = pd.DataFrame(calendar_rows)

    calendar_df.to_csv(
        CALENDAR_FILE,
        index=False,
    )

    print(f"\nSaved earnings calendar snapshot: {CALENDAR_FILE}")


def append_alert_history(
    alerts: list[dict],
) -> None:
    """
    保存本次已发送提醒。

    同一 ticker + earnings_date 只保留一条记录。
    """

    if not alerts:
        return

    HISTORY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = load_alert_history()
    new_rows = pd.DataFrame(alerts)

    combined = pd.concat(
        [existing, new_rows],
        ignore_index=True,
    )

    combined = combined.drop_duplicates(
        subset=["ticker", "earnings_date"],
        keep="last",
    )

    combined.to_csv(
        HISTORY_FILE,
        index=False,
    )

    print(f"\nUpdated earnings alert history: {HISTORY_FILE}")


# ============================================================
# 邮件内容
# ============================================================

def build_email_body(alerts: list[dict]) -> str:
    """
    生成 earnings 提醒邮件正文。
    """

    lines = [
        "以下公司明天有 earnings：",
        "",
    ]

    for alert in alerts:
        lines.append(
            f"- {alert['ticker']} ({alert['company']}): "
            f"earnings date = {alert['earnings_date']}"
        )

    lines.extend(
        [
            "",
            "This alert was generated by Investment OS.",
        ]
    )

    return "\n".join(lines)


# ============================================================
# 核心逻辑
# ============================================================

def find_tomorrow_earnings(
    companies: pd.DataFrame,
    target_date: date,
) -> tuple[list[dict], list[dict]]:
    """
    找出 target_date 发布财报、且还没有提醒过的公司。

    返回：
        alerts:
            需要发送提醒的公司列表。

        calendar_rows:
            当前查询到的 earnings calendar 快照。
    """

    history = load_alert_history()
    alerts: list[dict] = []
    calendar_rows: list[dict] = []

    current_date = today_et()

    for _, company_row in companies.iterrows():
        ticker = str(company_row["ticker"]).strip().upper()
        company = company_row.get("company", "")

        if not ticker:
            continue

        print(f"Checking earnings calendar for {ticker}...")

        earnings_dates = get_earnings_dates(ticker)

        earnings_dates_str = ";".join(
            earnings_date.isoformat()
            for earnings_date in earnings_dates
        )

        calendar_rows.append(
            {
                "date": current_date.isoformat(),
                "ticker": ticker,
                "company": company,
                "earnings_dates": earnings_dates_str,
                "updated_at": datetime.now(MARKET_TIMEZONE).strftime(
                    "%Y-%m-%d %H:%M:%S %Z"
                ),
            }
        )

        if target_date not in earnings_dates:
            continue

        if already_alerted(
            history,
            ticker,
            target_date,
        ):
            print(f"Already alerted for {ticker} earnings on {target_date}.")
            continue

        alerts.append(
            {
                "alert_date": current_date.isoformat(),
                "ticker": ticker,
                "company": company,
                "earnings_date": target_date.isoformat(),
                "sent_at": datetime.now(MARKET_TIMEZONE).strftime(
                    "%Y-%m-%d %H:%M:%S %Z"
                ),
            }
        )

    return alerts, calendar_rows


# ============================================================
# 主程序
# ============================================================

def main() -> None:
    """
    主程序入口。
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find input file: {INPUT_FILE}"
        )

    companies = pd.read_csv(INPUT_FILE)

    if "ticker" not in companies.columns:
        raise ValueError(
            "company_master.csv must contain a 'ticker' column"
        )

    target_date = today_et() + timedelta(days=1)

    print(f"\nChecking earnings calendar for: {target_date}\n")

    alerts, calendar_rows = find_tomorrow_earnings(
        companies,
        target_date,
    )

    # 无论有没有提醒，都保存当前 earnings calendar 快照。
    save_calendar_snapshot(calendar_rows)

    if not alerts:
        print("No earnings alerts for tomorrow.")
        return

    subject = f"Investment OS Earnings Alert: {target_date}"
    body = build_email_body(alerts)

    send_email(
        subject=subject,
        body=body,
    )

    append_alert_history(alerts)

    print(f"Sent {len(alerts)} earnings alert(s).")


if __name__ == "__main__":
    main()
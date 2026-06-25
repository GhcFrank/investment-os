"""
Send a weekly News Discovery summary email.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from news.fetch_semiengineering_news import FETCH_LOG_COLUMNS
from news.news_filter import (
    MANUAL_DECISION_COLUMNS,
    NEWS_COLUMNS,
    REJECT_COLUMNS,
    REVIEW_COLUMNS,
)
from news.news_utils import read_csv_safe
from utils.send_email import send_email


BASE_DIR = Path(__file__).resolve().parents[2]
NEWS_DIR = BASE_DIR / "data" / "news"

CURRENT_NEWS_FILE = NEWS_DIR / "semiengineering_news.csv"
HISTORY_FILE = NEWS_DIR / "semiengineering_news_history.csv"
REVIEW_FILE = NEWS_DIR / "news_review_queue.csv"
REJECT_FILE = NEWS_DIR / "news_rejected_log.csv"
MANUAL_DECISIONS_FILE = NEWS_DIR / "news_manual_decisions.csv"
FETCH_LOG_FILE = NEWS_DIR / "news_fetch_log.csv"

SUBJECT = "Investment OS Weekly News Update"


def _count_rows(path: Path, columns: list[str]) -> int:
    return len(read_csv_safe(path, columns))


def _clean_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    return str(value).strip()


def _latest_fetch(fetch_log: pd.DataFrame) -> dict[str, str]:
    if fetch_log.empty:
        return {}

    sort_columns = [
        column for column in ["completed_at", "started_at", "run_id"] if column in fetch_log.columns
    ]
    latest = fetch_log.copy()
    if sort_columns:
        latest = latest.sort_values(by=sort_columns, kind="stable")

    row = latest.tail(1).iloc[0]
    return {
        "run_at": _clean_value(row.get("completed_at"))
        or _clean_value(row.get("started_at")),
        "fetched_count": _clean_value(row.get("items_fetched")),
        "keep_count": _clean_value(row.get("keep_count")),
        "review_count": _clean_value(row.get("review_count")),
        "reject_count": _clean_value(row.get("reject_count")),
        "pages_requested": _clean_value(row.get("pages_requested")),
        "status": _clean_value(row.get("status")),
    }


def _top_review_items(review_queue: pd.DataFrame, limit: int = 10) -> list[str]:
    if review_queue.empty:
        return []

    rows = review_queue.head(limit)
    items: list[str] = []
    for _, row in rows.iterrows():
        published_at = _clean_value(row.get("published_at_gmt"))
        title = _clean_value(row.get("title"))
        items.append(f"- {published_at} | {title}")

    return items


def build_email_body() -> str:
    current_count = _count_rows(CURRENT_NEWS_FILE, NEWS_COLUMNS)
    history_count = _count_rows(HISTORY_FILE, NEWS_COLUMNS)
    review_queue = read_csv_safe(REVIEW_FILE, REVIEW_COLUMNS)
    reject_count = _count_rows(REJECT_FILE, REJECT_COLUMNS)
    manual_count = _count_rows(MANUAL_DECISIONS_FILE, MANUAL_DECISION_COLUMNS)
    fetch_log = read_csv_safe(FETCH_LOG_FILE, FETCH_LOG_COLUMNS)
    latest_fetch = _latest_fetch(fetch_log)

    lines = [
        "Weekly News Discovery completed.",
        "",
        "Lookback window: 7 days",
        "",
        f"Current keep count: {current_count}",
        f"Total keep history count: {history_count}",
        f"Review queue count: {len(review_queue)}",
        f"Rejected log count: {reject_count}",
        f"Manual decisions count: {manual_count}",
        "",
        "Latest fetch:",
        f"- run_at: {latest_fetch.get('run_at', '')}",
        f"- fetched_count: {latest_fetch.get('fetched_count', '')}",
        f"- keep_count: {latest_fetch.get('keep_count', '')}",
        f"- review_count: {latest_fetch.get('review_count', '')}",
        f"- reject_count: {latest_fetch.get('reject_count', '')}",
        f"- pages_requested: {latest_fetch.get('pages_requested', '')}",
        f"- status: {latest_fetch.get('status', '')}",
        "",
        "Please review:",
        "data/news/news_review_queue.csv",
        "",
        "To record manual decisions, edit:",
        "data/news/news_manual_decisions.csv",
    ]

    top_items = _top_review_items(review_queue)
    if top_items:
        lines.extend(["", "Top review queue items:", *top_items])

    return "\n".join(lines)


def main() -> None:
    send_email(subject=SUBJECT, body=build_email_body())
    print("Weekly news email sent successfully.")


if __name__ == "__main__":
    main()

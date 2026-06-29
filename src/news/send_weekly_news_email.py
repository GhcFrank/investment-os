"""
Send a weekly News Discovery summary email.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from news.fetch_semiengineering_news import FETCH_LOG_COLUMNS
from news.fetch_semiengineering_news import LATEST_FETCH_MANIFEST_COLUMNS
from news.deepseek_news_analyzer import ANALYSIS_COLUMNS
from news.analyze_news_with_deepseek import LOG_COLUMNS as DEEPSEEK_LOG_COLUMNS
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
LATEST_FETCH_MANIFEST_FILE = NEWS_DIR / "news_latest_fetch_manifest.csv"
DEEPSEEK_ANALYSIS_FILE = NEWS_DIR / "deepseek_news_analysis.csv"
DEEPSEEK_ANALYSIS_LOG_FILE = NEWS_DIR / "deepseek_news_analysis_log.csv"

FLASH_MODEL = "deepseek-v4-flash"
FLASH_PROMPT_VERSION = "deepseek_flash_preprocess_v1"

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


def _decision_counts(rows: pd.DataFrame) -> list[str]:
    if rows.empty:
        return ["- keep: 0", "- watch: 0", "- reject: 0", "- failed: 0"]

    decisions = rows["recommended_decision"].fillna("").astype(str).str.lower()
    statuses = rows["status"].fillna("").astype(str).str.lower()
    return [
        f"- keep: {int(((statuses == 'ok') & (decisions == 'keep')).sum())}",
        f"- watch: {int(((statuses == 'ok') & (decisions == 'watch')).sum())}",
        f"- reject: {int(((statuses == 'ok') & (decisions == 'reject')).sum())}",
        f"- failed: {int((statuses != 'ok').sum())}",
    ]


def _top_flash_items(rows: pd.DataFrame, limit: int = 10) -> list[str]:
    if rows.empty:
        return ["- none"]

    work = rows.copy()
    work["recommended_decision"] = (
        work["recommended_decision"].fillna("").astype(str).str.lower()
    )
    work = work[work["recommended_decision"].isin(["keep", "watch"])].copy()
    if work.empty:
        return ["- none"]

    work["_relevance"] = pd.to_numeric(work["relevance_score"], errors="coerce").fillna(0)
    work["_impact"] = pd.to_numeric(work["impact_score"], errors="coerce").fillna(0)
    work = work.sort_values(
        by=["_relevance", "_impact", "news_id"],
        ascending=[False, False, True],
        kind="stable",
    )

    items: list[str] = []
    for _, row in work.head(limit).iterrows():
        title = _clean_value(row.get("title")) or _clean_value(row.get("news_id"))
        items.extend(
            [
                f"- {title}",
                f"  url: {_clean_value(row.get('url'))}",
                f"  recommended_decision: {_clean_value(row.get('recommended_decision'))}",
                f"  relevance_score: {_clean_value(row.get('relevance_score'))}",
                f"  impact_score: {_clean_value(row.get('impact_score'))}",
                f"  primary_tickers: {_clean_value(row.get('primary_tickers'))}",
                f"  secondary_tickers: {_clean_value(row.get('secondary_tickers'))}",
                f"  primary_subthemes: {_clean_value(row.get('primary_subthemes'))}",
                f"  summary: {_clean_value(row.get('summary'))}",
                f"  why_it_matters: {_clean_value(row.get('why_it_matters'))}",
            ]
        )

    return items


def _deepseek_flash_review_lines() -> list[str]:
    manifest = read_csv_safe(LATEST_FETCH_MANIFEST_FILE, LATEST_FETCH_MANIFEST_COLUMNS)
    analysis = read_csv_safe(DEEPSEEK_ANALYSIS_FILE, ANALYSIS_COLUMNS)
    log = read_csv_safe(DEEPSEEK_ANALYSIS_LOG_FILE, DEEPSEEK_LOG_COLUMNS)

    lines = ["", "DeepSeek Flash News Review"]
    if manifest.empty or analysis.empty or "news_id" not in manifest.columns:
        return [
            *lines,
            "DeepSeek Flash review was skipped or no new articles were eligible.",
        ]

    manifest_ids = set(manifest["news_id"].fillna("").astype(str).str.strip())
    manifest_ids.discard("")
    if not manifest_ids:
        return [
            *lines,
            "DeepSeek Flash review was skipped or no new articles were eligible.",
        ]

    latest_log = log[
        (log["model"].fillna("").astype(str) == FLASH_MODEL)
        & (log["prompt_version"].fillna("").astype(str) == FLASH_PROMPT_VERSION)
    ].copy()
    if latest_log.empty:
        log_row = {}
    else:
        latest_log = latest_log.sort_values(by=["run_at"], kind="stable")
        log_row = latest_log.tail(1).iloc[0].to_dict()

    related = analysis[
        (analysis["model"].fillna("").astype(str) == FLASH_MODEL)
        & (analysis["prompt_version"].fillna("").astype(str) == FLASH_PROMPT_VERSION)
        & (analysis["news_id"].fillna("").astype(str).isin(manifest_ids))
    ].copy()
    if related.empty:
        return [
            *lines,
            "DeepSeek Flash review was skipped or no new articles were eligible.",
        ]

    title_map = manifest.drop_duplicates(subset=["news_id"], keep="last").set_index("news_id")
    related["title"] = related["news_id"].map(title_map["title"]).fillna("")
    related["url"] = related["news_id"].map(title_map["url"]).fillna("")

    lines.extend(
        [
            "Analyzed this run:",
            f"- analyzed: {_clean_value(log_row.get('articles_analyzed', ''))}",
            f"- skipped: {_clean_value(log_row.get('articles_skipped', ''))}",
            f"- failed: {_clean_value(log_row.get('articles_failed', ''))}",
            "",
            "Decision counts:",
            *_decision_counts(related),
            "",
            "Top keep/watch:",
            *_top_flash_items(related),
        ]
    )
    return lines


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

    lines.extend(_deepseek_flash_review_lines())

    return "\n".join(lines)


def main() -> None:
    send_email(subject=SUBJECT, body=build_email_body())
    print("Weekly news email sent successfully.")


if __name__ == "__main__":
    main()

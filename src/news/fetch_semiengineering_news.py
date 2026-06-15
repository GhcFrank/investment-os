"""
Backfill Semiconductor Engineering news through the WordPress REST API.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from news.news_filter import (
    NEWS_COLUMNS,
    REJECT_COLUMNS,
    REVIEW_COLUMNS,
    load_filter_config,
    rows_from_posts,
    upsert_keep_history,
    upsert_rejected_log,
    upsert_review_queue,
)
from news.news_utils import (
    atomic_write_csv,
    format_utc,
    get_json,
    now_utc_iso,
    parse_utc_datetime,
    read_csv_safe,
)


BASE_DIR = Path(__file__).resolve().parents[2]
NEWS_DIR = BASE_DIR / "data" / "news"
RAW_DIR = NEWS_DIR / "raw" / "semiengineering"
REFERENCE_DIR = NEWS_DIR / "reference"

SOURCE_ID = "semiengineering"
API_BASE = "https://semiengineering.com/wp-json/wp/v2"
POSTS_ENDPOINT = f"{API_BASE}/posts"
CATEGORIES_ENDPOINT = f"{API_BASE}/categories"
TAGS_ENDPOINT = f"{API_BASE}/tags"

CURRENT_NEWS_FILE = NEWS_DIR / "semiengineering_news.csv"
HISTORY_FILE = NEWS_DIR / "semiengineering_news_history.csv"
REVIEW_FILE = NEWS_DIR / "news_review_queue.csv"
REJECT_FILE = NEWS_DIR / "news_rejected_log.csv"
FETCH_LOG_FILE = NEWS_DIR / "news_fetch_log.csv"
CATEGORIES_FILE = REFERENCE_DIR / "semiengineering_categories.csv"
TAGS_FILE = REFERENCE_DIR / "semiengineering_tags.csv"

TERM_COLUMNS = [
    "term_id",
    "name",
    "slug",
    "count",
    "updated_at",
]

FETCH_LOG_COLUMNS = [
    "run_id",
    "started_at",
    "completed_at",
    "source_id",
    "mode",
    "window_start",
    "window_end",
    "pages_requested",
    "items_fetched",
    "keep_count",
    "review_count",
    "reject_count",
    "raw_file",
    "status",
    "error_message",
]

POST_FIELDS = (
    "id,date,date_gmt,modified,modified_gmt,slug,link,"
    "title,excerpt,content,author,categories,tags"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill Semiconductor Engineering news."
    )
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--save-raw", dest="save_raw", action="store_true")
    parser.add_argument("--no-save-raw", dest="save_raw", action="store_false")
    parser.set_defaults(save_raw=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reprocess-raw", default=None)

    return parser.parse_args()


def compact_timestamp(value: str) -> str:
    """
    Convert a UTC ISO string into YYYYMMDDTHHMMSSZ.
    """

    return value.replace("-", "").replace(":", "")


def fetch_paginated(
    endpoint: str,
    params: dict[str, object],
    sleep_seconds: float = 0.5,
    label: str = "endpoint",
) -> tuple[list[dict[str, Any]], int, int, int]:
    """
    Fetch a paginated WordPress REST endpoint.
    """

    page = 1
    total_pages = 1
    total_items = 0
    pages_requested = 0
    items: list[dict[str, Any]] = []

    while page <= total_pages:
        request_params = {
            **params,
            "page": page,
        }
        response = get_json(endpoint, params=request_params)
        pages_requested += 1
        payload = response.json()

        if not isinstance(payload, list):
            raise ValueError(f"Expected list response from {endpoint}")

        if page == 1:
            total_pages = int(response.headers.get("X-WP-TotalPages", "1") or 1)
            total_items = int(response.headers.get("X-WP-Total", "0") or 0)
            print(
                f"Fetching {label}: {total_items} item(s) across {total_pages} page(s)",
                flush=True,
            )

        items.extend(payload)

        if page == total_pages or page % 50 == 0:
            print(f"Fetched {label} page {page}/{total_pages}", flush=True)

        if page < total_pages:
            time.sleep(sleep_seconds)

        page += 1

    return items, pages_requested, total_pages, total_items


def fetch_terms(endpoint: str) -> tuple[list[dict[str, Any]], int]:
    """
    Fetch category or tag reference terms.
    """

    terms, pages_requested, _, _ = fetch_paginated(
        endpoint,
        {
            "per_page": 100,
            "orderby": "name",
            "order": "asc",
            "_fields": "id,name,slug,count",
        },
        label=endpoint.rsplit("/", 1)[-1],
    )

    return terms, pages_requested


def term_rows(terms: list[dict[str, Any]], updated_at: str) -> pd.DataFrame:
    rows = [
        {
            "term_id": str(term.get("id", "") or ""),
            "name": str(term.get("name", "") or ""),
            "slug": str(term.get("slug", "") or ""),
            "count": str(term.get("count", "") or ""),
            "updated_at": updated_at,
        }
        for term in terms
    ]

    return pd.DataFrame(rows, columns=TERM_COLUMNS)


def build_term_map(terms: list[dict[str, Any]]) -> dict[int, str]:
    term_map: dict[int, str] = {}

    for term in terms:
        try:
            term_id = int(term.get("id"))
        except (TypeError, ValueError):
            continue

        name = str(term.get("name", "") or "").strip()

        if name:
            term_map[term_id] = name

    return term_map


def load_reference_map(path: Path) -> dict[int, str]:
    df = read_csv_safe(path, TERM_COLUMNS)
    term_map: dict[int, str] = {}

    for _, row in df.iterrows():
        try:
            term_id = int(row.get("term_id", ""))
        except (TypeError, ValueError):
            continue

        name = str(row.get("name", "") or "").strip()

        if name:
            term_map[term_id] = name

    return term_map


def fetch_posts(
    window_start: str,
    window_end: str,
) -> tuple[list[dict[str, Any]], int, int, int]:
    """
    Fetch Semiconductor Engineering posts for a UTC time window.
    """

    return fetch_paginated(
        POSTS_ENDPOINT,
        {
            "after": window_start,
            "before": window_end,
            "per_page": 100,
            "orderby": "date",
            "order": "asc",
            "_fields": POST_FIELDS,
        },
        label="posts",
    )


def save_raw_payload(
    posts: list[dict[str, Any]],
    fetched_at: str,
    window_start: str,
    window_end: str,
    total_pages: int,
    total_items: int,
) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_file = RAW_DIR / f"backfill_{compact_timestamp(window_end)}.json"
    payload = {
        "source_id": SOURCE_ID,
        "fetched_at": fetched_at,
        "window_start": window_start,
        "window_end": window_end,
        "api_endpoint": POSTS_ENDPOINT,
        "total_pages": total_pages,
        "total_items": total_items,
        "posts": posts,
    }
    temp_file = raw_file.with_name(f".{raw_file.name}.tmp")
    temp_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_file.replace(raw_file)

    return raw_file


def load_raw_payload(raw_file: Path) -> dict[str, Any]:
    payload = json.loads(raw_file.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError(f"Raw payload must be a JSON object: {raw_file}")

    posts = payload.get("posts", [])

    if not isinstance(posts, list):
        raise ValueError(f"Raw payload posts must be a list: {raw_file}")

    return payload


def append_fetch_log(row: dict[str, object]) -> None:
    existing = read_csv_safe(FETCH_LOG_FILE, FETCH_LOG_COLUMNS)
    new_row = pd.DataFrame([row], columns=FETCH_LOG_COLUMNS)
    combined = pd.concat([existing, new_row], ignore_index=True)
    atomic_write_csv(combined, FETCH_LOG_FILE, FETCH_LOG_COLUMNS)


def split_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keep = rows[rows["filter_status"] == "keep"].copy()
    review = rows[rows["filter_status"] == "review"].copy()
    reject = rows[rows["filter_status"] == "reject"].copy()

    return keep, review, reject


def write_outputs(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keep, review, reject = split_rows(rows)

    atomic_write_csv(keep, CURRENT_NEWS_FILE, NEWS_COLUMNS)
    upsert_keep_history(keep, HISTORY_FILE)
    upsert_review_queue(review, REVIEW_FILE)
    upsert_rejected_log(reject, REJECT_FILE)

    return keep, review, reject


def count_pipe_values(series: pd.Series) -> dict[str, int]:
    counts: dict[str, int] = {}

    for value in series.dropna().astype(str):
        for item in value.split("|"):
            item = item.strip()

            if item:
                counts[item] = counts.get(item, 0) + 1

    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def print_top_counts(title: str, counts: dict[str, int], limit: int = 10) -> None:
    print(title)

    if not counts:
        print("- none")
        return

    for item, count in list(counts.items())[:limit]:
        print(f"{item}: {count}")


def print_titles(title: str, rows: pd.DataFrame, limit: int = 10) -> None:
    print(title)

    if rows.empty:
        print("- none")
        return

    for _, row in rows.head(limit).iterrows():
        print(f"- {row['title']}")


def print_summary(
    window_start: str,
    window_end: str,
    rows: pd.DataFrame,
    keep: pd.DataFrame,
    review: pd.DataFrame,
    reject: pd.DataFrame,
    raw_file: str,
) -> None:
    emerging_count = int((rows["emerging_candidate"].astype(str) == "True").sum())

    print("\nSemiconductor Engineering News Backfill\n")
    print("Window:")
    print(window_start)
    print("to")
    print(window_end)
    print()
    print(f"Fetched: {len(rows)}")
    print(f"Keep: {len(keep)}")
    print(f"Review: {len(review)}")
    print(f"Reject: {len(reject)}")
    print()
    print_top_counts("Top matched themes:", count_pipe_values(keep["matched_subthemes"]))
    print()
    print_top_counts("Direct company matches:", count_pipe_values(keep["matched_tickers"]))
    print()
    print(f"Emerging candidates: {emerging_count}")
    print()
    print(f"Saved raw:\n{raw_file or '(not saved)'}")
    print(f"\nSaved keep history:\n{HISTORY_FILE}")
    print(f"\nSaved review queue:\n{REVIEW_FILE}")
    print(f"\nSaved reject log:\n{REJECT_FILE}")
    print(f"\nSaved fetch log:\n{FETCH_LOG_FILE}")
    print()
    print_titles("Top 10 kept titles", keep)
    print()
    print_titles("Top 10 review titles", review)


def get_window(args: argparse.Namespace) -> tuple[str, str]:
    if args.as_of:
        window_end_dt = parse_utc_datetime(args.as_of)
    else:
        window_end_dt = datetime.now(UTC).replace(microsecond=0)

    window_start_dt = window_end_dt - timedelta(days=args.lookback_days)

    return format_utc(window_start_dt), format_utc(window_end_dt)


def run_backfill(args: argparse.Namespace) -> int:
    started_at = now_utc_iso()
    completed_at = ""
    run_id = f"news_backfill_{compact_timestamp(started_at)}"
    mode = "dry_run" if args.dry_run else "backfill"
    pages_requested = 0
    raw_file = ""
    taxonomy_warning = ""
    window_start = ""
    window_end = ""
    rows = pd.DataFrame(columns=NEWS_COLUMNS)
    keep = pd.DataFrame(columns=NEWS_COLUMNS)
    review = pd.DataFrame(columns=NEWS_COLUMNS)
    reject = pd.DataFrame(columns=NEWS_COLUMNS)

    try:
        if args.reprocess_raw:
            mode = "reprocess_raw_dry_run" if args.dry_run else "reprocess_raw"
            raw_path = Path(args.reprocess_raw)
            payload = load_raw_payload(raw_path)
            posts = payload["posts"]
            window_start = str(payload.get("window_start", ""))
            window_end = str(payload.get("window_end", ""))
            raw_file = str(raw_path)
            category_map = load_reference_map(CATEGORIES_FILE)
            tag_map = load_reference_map(TAGS_FILE)
            total_pages = int(payload.get("total_pages", 0) or 0)
            total_items = int(payload.get("total_items", len(posts)) or len(posts))

        else:
            window_start, window_end = get_window(args)
            category_terms: list[dict[str, Any]] = []
            tag_terms: list[dict[str, Any]] = []
            category_map: dict[int, str] = {}
            tag_map: dict[int, str] = {}

            try:
                category_terms, category_pages = fetch_terms(CATEGORIES_ENDPOINT)
                tag_terms, tag_pages = fetch_terms(TAGS_ENDPOINT)
                pages_requested += category_pages + tag_pages
                category_map = build_term_map(category_terms)
                tag_map = build_term_map(tag_terms)

                if not args.dry_run:
                    updated_at = now_utc_iso()
                    atomic_write_csv(
                        term_rows(category_terms, updated_at),
                        CATEGORIES_FILE,
                        TERM_COLUMNS,
                    )
                    atomic_write_csv(
                        term_rows(tag_terms, updated_at),
                        TAGS_FILE,
                        TERM_COLUMNS,
                    )

            except Exception as error:
                taxonomy_warning = f"taxonomy warning: {error}"
                print(f"Warning: {taxonomy_warning}")

            posts, post_pages, total_pages, total_items = fetch_posts(
                window_start=window_start,
                window_end=window_end,
            )
            pages_requested += post_pages

            if args.save_raw and not args.dry_run:
                raw_path = save_raw_payload(
                    posts=posts,
                    fetched_at=now_utc_iso(),
                    window_start=window_start,
                    window_end=window_end,
                    total_pages=total_pages,
                    total_items=total_items,
                )
                raw_file = str(raw_path)

        if not posts:
            raise RuntimeError("No Semiconductor Engineering posts were fetched.")

        config = load_filter_config(BASE_DIR)
        seen_at = now_utc_iso()
        rows = rows_from_posts(
            posts=posts,
            category_map=category_map,
            tag_map=tag_map,
            seen_at=seen_at,
            config=config,
        )
        keep, review, reject = split_rows(rows)

        if not args.dry_run:
            keep, review, reject = write_outputs(rows)

        completed_at = now_utc_iso()
        error_message = taxonomy_warning

        append_fetch_log(
            {
                "run_id": run_id,
                "started_at": started_at,
                "completed_at": completed_at,
                "source_id": SOURCE_ID,
                "mode": mode,
                "window_start": window_start,
                "window_end": window_end,
                "pages_requested": pages_requested,
                "items_fetched": len(rows),
                "keep_count": len(keep),
                "review_count": len(review),
                "reject_count": len(reject),
                "raw_file": raw_file,
                "status": "success",
                "error_message": error_message,
            }
        )

        print_summary(
            window_start=window_start,
            window_end=window_end,
            rows=rows,
            keep=keep,
            review=review,
            reject=reject,
            raw_file=raw_file,
        )

        _ = total_pages
        _ = total_items
        return 0

    except Exception as error:
        completed_at = now_utc_iso()
        error_message = str(error)

        try:
            append_fetch_log(
                {
                    "run_id": run_id,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "source_id": SOURCE_ID,
                    "mode": mode,
                    "window_start": window_start,
                    "window_end": window_end,
                    "pages_requested": pages_requested,
                    "items_fetched": len(rows),
                    "keep_count": len(keep),
                    "review_count": len(review),
                    "reject_count": len(reject),
                    "raw_file": raw_file,
                    "status": "failed",
                    "error_message": error_message,
                }
            )
        except Exception as log_error:
            print(f"Warning: failed to write fetch log: {log_error}")

        print(f"Error: {error_message}", file=sys.stderr)
        return 1


def main() -> None:
    sys.exit(run_backfill(parse_args()))


if __name__ == "__main__":
    main()

"""
Backfill Semiconductor Engineering news through the WordPress REST API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from news.news_filter import (
    MANUAL_DECISION_COLUMNS,
    MANUAL_DECISION_DATE_COLUMNS,
    NEWS_COLUMNS,
    NEWS_DATE_COLUMNS,
    REJECT_COLUMNS,
    REVIEW_COLUMNS,
    load_manual_decisions,
    load_filter_config,
    normalize_news_date_columns,
    reconcile_news_statuses,
    rows_from_posts,
    split_rows,
)
from news.news_utils import (
    atomic_write_csv,
    clean_html_text,
    format_utc,
    get_json,
    now_utc_iso,
    parse_utc_datetime,
    read_csv_safe,
    repo_relative_path,
    to_yyyy_mm_dd,
    utc_today_yyyy_mm_dd,
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
MANUAL_DECISIONS_FILE = NEWS_DIR / "news_manual_decisions.csv"
LATEST_FETCH_MANIFEST_FILE = NEWS_DIR / "news_latest_fetch_manifest.csv"
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

LATEST_FETCH_MANIFEST_COLUMNS = [
    "fetch_run_at",
    "news_id",
    "title",
    "url",
    "published_at",
    "updated_at",
    "change_status",
    "source",
]

POST_FIELDS = (
    "id,date,date_gmt,modified,modified_gmt,slug,link,"
    "title,excerpt,author,categories,tags"
)

DEFAULT_LOOKBACK_DAYS = 30


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill Semiconductor Engineering news."
    )
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--save-raw", dest="save_raw", action="store_true")
    parser.add_argument("--no-save-raw", dest="save_raw", action="store_false")
    parser.set_defaults(save_raw=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reprocess-raw", default=None)
    parser.add_argument("--apply-review-decisions", action="store_true")
    args = parser.parse_args(argv)
    validate_cli_args(args, parser)

    return args


def validate_cli_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser | None = None,
) -> None:
    def fail(message: str) -> None:
        if parser is not None:
            parser.error(message)

        raise ValueError(message)

    lookback_days = getattr(args, "lookback_days", DEFAULT_LOOKBACK_DAYS)
    apply_review = bool(getattr(args, "apply_review_decisions", False))
    reprocess_raw = bool(getattr(args, "reprocess_raw", None))
    dry_run = bool(getattr(args, "dry_run", False))
    as_of = getattr(args, "as_of", None)

    if apply_review and dry_run:
        fail("--apply-review-decisions cannot be combined with --dry-run")

    if apply_review and reprocess_raw:
        fail("--apply-review-decisions cannot be combined with --reprocess-raw")

    if apply_review and as_of:
        fail("--apply-review-decisions cannot be combined with --as-of")

    if apply_review and lookback_days != DEFAULT_LOOKBACK_DAYS:
        fail("--apply-review-decisions cannot be combined with --lookback-days")

    if reprocess_raw and as_of:
        fail("--reprocess-raw cannot be combined with --as-of")

    if reprocess_raw and lookback_days != DEFAULT_LOOKBACK_DAYS:
        fail("--reprocess-raw cannot be combined with --lookback-days")


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


def collect_post_term_ids(
    posts: list[dict[str, Any]],
) -> tuple[set[int], set[int]]:
    """
    Collect category and tag IDs actually used by the fetched posts.
    """

    category_ids: set[int] = set()
    tag_ids: set[int] = set()

    for post in posts:
        for value in post.get("categories", []) or []:
            try:
                category_ids.add(int(value))
            except (TypeError, ValueError):
                continue

        for value in post.get("tags", []) or []:
            try:
                tag_ids.add(int(value))
            except (TypeError, ValueError):
                continue

    return category_ids, tag_ids


def fetch_terms_by_ids(
    endpoint: str,
    term_ids: set[int],
    batch_size: int = 100,
) -> tuple[list[dict[str, Any]], int]:
    """
    Fetch taxonomy terms by ID batches.
    """

    if not term_ids:
        return [], 0

    sorted_ids = sorted(term_ids)
    all_terms: list[dict[str, Any]] = []
    pages_requested = 0

    for start in range(0, len(sorted_ids), batch_size):
        batch = sorted_ids[start : start + batch_size]
        terms, batch_pages, _, _ = fetch_paginated(
            endpoint,
            {
                "include": ",".join(str(term_id) for term_id in batch),
                "per_page": 100,
                "_fields": "id,name,slug,count",
            },
            label=f"{endpoint.rsplit('/', 1)[-1]} include {start + 1}-{start + len(batch)}",
        )
        pages_requested += batch_pages
        all_terms.extend(terms)

    return all_terms, pages_requested


def term_rows(terms: list[dict[str, Any]], updated_at: str) -> pd.DataFrame:
    rows = [
        {
            "term_id": str(term.get("id", "") or ""),
            "name": clean_html_text(term.get("name", "")),
            "slug": clean_html_text(term.get("slug", "")),
            "count": str(term.get("count", "") or ""),
            "updated_at": updated_at,
        }
        for term in terms
    ]

    return pd.DataFrame(rows, columns=TERM_COLUMNS)


def upsert_term_cache(
    existing: pd.DataFrame,
    new_terms: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge newly fetched taxonomy terms into the local cache.
    """

    combined = pd.concat(
        [
            existing.reindex(columns=TERM_COLUMNS),
            new_terms.reindex(columns=TERM_COLUMNS),
        ],
        ignore_index=True,
    )

    if combined.empty:
        return pd.DataFrame(columns=TERM_COLUMNS)

    combined["term_id"] = combined["term_id"].fillna("").astype(str)
    combined = combined[combined["term_id"] != ""].copy()
    combined = combined.drop_duplicates(subset=["term_id"], keep="last")
    combined["_term_id_num"] = pd.to_numeric(
        combined["term_id"],
        errors="coerce",
    )
    combined = combined.sort_values(
        by=["_term_id_num", "term_id"],
        na_position="last",
    ).drop(columns=["_term_id_num"])

    return combined.reindex(columns=TERM_COLUMNS)


def build_term_map(terms: pd.DataFrame | list[dict[str, Any]]) -> dict[int, str]:
    """
    Build an ID-to-name map from term rows.
    """

    if isinstance(terms, list):
        df = term_rows(terms, now_utc_iso())
    else:
        df = terms

    term_map: dict[int, str] = {}

    for _, row in df.iterrows():
        try:
            term_id = int(row.get("term_id", ""))
        except (TypeError, ValueError):
            try:
                term_id = int(row.get("id", ""))
            except (TypeError, ValueError):
                continue

        name = clean_html_text(row.get("name", ""))

        if name:
            term_map[term_id] = name

    return term_map


def load_reference_cache(path: Path) -> pd.DataFrame:
    """
    Load taxonomy reference cache.
    """

    cache = read_csv_safe(path, TERM_COLUMNS)
    cache["name"] = cache["name"].map(clean_html_text)
    cache["slug"] = cache["slug"].map(clean_html_text)

    return cache


def load_reference_map(path: Path) -> dict[int, str]:
    return build_term_map(load_reference_cache(path))


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


def _raw_safe_post(post: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in post.items()
        if key != "content"
    }


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
        "posts": [_raw_safe_post(post) for post in posts],
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


def _clean_manifest_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def content_hash(row: pd.Series | dict[str, object]) -> str:
    """
    Stable hash over article fields that indicate meaningful content changes.
    """

    parts = [
        _clean_manifest_value(row.get("title", "")),
        _clean_manifest_value(row.get("summary", "")),
        _clean_manifest_value(row.get("excerpt", "")),
        _clean_manifest_value(row.get("url", "")),
        _clean_manifest_value(row.get("published_at_gmt", "")),
        _clean_manifest_value(row.get("modified_at_gmt", "")),
    ]
    payload = "\n".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _latest_rows_by_news_id(frames: list[pd.DataFrame]) -> dict[str, pd.Series]:
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if combined.empty or "news_id" not in combined.columns:
        return {}

    combined["news_id"] = combined["news_id"].fillna("").astype(str).str.strip()
    combined = combined[combined["news_id"] != ""].copy()
    if combined.empty:
        return {}

    return {
        str(row.get("news_id", "") or ""): row
        for _, row in combined.drop_duplicates(subset=["news_id"], keep="last").iterrows()
    }


def _row_changed(current: pd.Series, old: pd.Series) -> bool:
    for field in [
        "modified_at_gmt",
        "content_hash",
        "title",
        "summary",
        "excerpt",
        "url",
        "published_at_gmt",
    ]:
        if field == "content_hash":
            current_value = content_hash(current)
            old_value = content_hash(old)
        elif field not in current.index or field not in old.index:
            continue
        else:
            current_value = _clean_manifest_value(current.get(field, ""))
            old_value = _clean_manifest_value(old.get(field, ""))

        if current_value and old_value and current_value != old_value:
            return True

    return False


def build_latest_fetch_manifest(
    rows: pd.DataFrame,
    previous_frames: list[pd.DataFrame],
    fetch_run_at: str,
) -> pd.DataFrame:
    """
    Return rows that are new or materially updated in the current fetch.
    """

    previous_by_id = _latest_rows_by_news_id(previous_frames)
    manifest_rows: list[dict[str, str]] = []

    if rows.empty or "news_id" not in rows.columns:
        return pd.DataFrame(columns=LATEST_FETCH_MANIFEST_COLUMNS)

    for _, row in rows.iterrows():
        news_id = _clean_manifest_value(row.get("news_id", ""))
        if not news_id:
            continue

        old = previous_by_id.get(news_id)
        if old is None:
            change_status = "new"
        elif _row_changed(row, old):
            change_status = "updated"
        else:
            continue

        manifest_rows.append(
            {
                "fetch_run_at": fetch_run_at,
                "news_id": news_id,
                "title": _clean_manifest_value(row.get("title", "")),
                "url": _clean_manifest_value(row.get("url", "")),
                "published_at": _clean_manifest_value(
                    row.get("published_at_gmt", row.get("published_at_local", ""))
                ),
                "updated_at": _clean_manifest_value(row.get("modified_at_gmt", "")),
                "change_status": change_status,
                "source": _clean_manifest_value(row.get("source_id", SOURCE_ID)) or SOURCE_ID,
            }
        )

    return pd.DataFrame(manifest_rows, columns=LATEST_FETCH_MANIFEST_COLUMNS)


def write_latest_fetch_manifest(manifest: pd.DataFrame) -> None:
    atomic_write_csv(
        manifest.reindex(columns=LATEST_FETCH_MANIFEST_COLUMNS),
        LATEST_FETCH_MANIFEST_FILE,
        LATEST_FETCH_MANIFEST_COLUMNS,
    )


def write_outputs(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = normalize_news_date_columns(rows)
    reconcile_news_statuses(
        rows=rows,
        history_path=HISTORY_FILE,
        review_path=REVIEW_FILE,
        reject_path=REJECT_FILE,
    )
    keep, review, reject = split_rows(rows)
    current_keep = rows[rows["filter_status"].isin(["keep", "manual_keep"])].copy()
    atomic_write_csv(current_keep, CURRENT_NEWS_FILE, NEWS_COLUMNS)

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
    print_top_counts(
        "Content class distribution:",
        count_pipe_values(rows["content_class"]),
    )
    print()
    print_top_counts(
        "Source quality distribution:",
        count_pipe_values(rows["source_quality"]),
    )
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


def _taxonomy_maps_for_posts(
    posts: list[dict[str, Any]],
    dry_run: bool,
) -> tuple[dict[int, str], dict[int, str], int, str]:
    """
    Load taxonomy cache, fetch only missing used terms, and return maps.
    """

    warning = ""
    pages_requested = 0
    category_ids, tag_ids = collect_post_term_ids(posts)
    category_cache = load_reference_cache(CATEGORIES_FILE)
    tag_cache = load_reference_cache(TAGS_FILE)
    known_categories = set(pd.to_numeric(category_cache["term_id"], errors="coerce").dropna().astype(int))
    known_tags = set(pd.to_numeric(tag_cache["term_id"], errors="coerce").dropna().astype(int))
    missing_category_ids = category_ids - known_categories
    missing_tag_ids = tag_ids - known_tags

    try:
        category_terms, category_pages = fetch_terms_by_ids(
            CATEGORIES_ENDPOINT,
            missing_category_ids,
        )
        tag_terms, tag_pages = fetch_terms_by_ids(
            TAGS_ENDPOINT,
            missing_tag_ids,
        )
        pages_requested += category_pages + tag_pages
        updated_at = now_utc_iso()
        category_cache = upsert_term_cache(
            category_cache,
            term_rows(category_terms, updated_at),
        )
        tag_cache = upsert_term_cache(
            tag_cache,
            term_rows(tag_terms, updated_at),
        )

        if not dry_run:
            atomic_write_csv(category_cache, CATEGORIES_FILE, TERM_COLUMNS)
            atomic_write_csv(tag_cache, TAGS_FILE, TERM_COLUMNS)

    except Exception as error:
        warning = f"taxonomy warning: {error}"
        print(f"Warning: {warning}")

    return build_term_map(category_cache), build_term_map(tag_cache), pages_requested, warning


def _fetch_and_classify(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, str, str, str, int, int]:
    """
    Fetch or load raw posts and return classified rows plus run metadata.
    """

    pages_requested = 0
    raw_file = ""

    if args.reprocess_raw:
        raw_path = Path(args.reprocess_raw)
        payload = load_raw_payload(raw_path)
        posts = payload["posts"]
        window_start = str(payload.get("window_start", ""))
        window_end = str(payload.get("window_end", ""))
        raw_file = repo_relative_path(raw_path, BASE_DIR)
        category_map = load_reference_map(CATEGORIES_FILE)
        tag_map = load_reference_map(TAGS_FILE)

    else:
        window_start, window_end = get_window(args)
        posts, post_pages, _, _ = fetch_posts(
            window_start=window_start,
            window_end=window_end,
        )
        pages_requested += post_pages
        category_map, tag_map, taxonomy_pages, taxonomy_warning = _taxonomy_maps_for_posts(
            posts,
            dry_run=args.dry_run,
        )
        pages_requested += taxonomy_pages

        if args.save_raw and not args.dry_run:
            raw_path = save_raw_payload(
                posts=posts,
                fetched_at=now_utc_iso(),
                window_start=window_start,
                window_end=window_end,
                total_pages=post_pages,
                total_items=len(posts),
            )
            raw_file = repo_relative_path(raw_path, BASE_DIR)

    if not posts:
        raise RuntimeError("No Semiconductor Engineering posts were fetched.")

    config = load_filter_config(BASE_DIR, MANUAL_DECISIONS_FILE)
    rows = rows_from_posts(
        posts=posts,
        category_map=category_map,
        tag_map=tag_map,
        seen_at=utc_today_yyyy_mm_dd(),
        config=config,
    )

    return rows, window_start, window_end, raw_file, pages_requested, len(posts)


def _state_row_by_id(frame: pd.DataFrame, news_id: str) -> pd.Series | None:
    if frame.empty or "news_id" not in frame.columns:
        return None

    matches = frame[frame["news_id"].fillna("").astype(str) == news_id]

    if matches.empty:
        return None

    return matches.iloc[-1]


def _news_row_from_state(row: pd.Series, applied_at: str) -> dict[str, object]:
    data = {column: "" for column in NEWS_COLUMNS}

    for column in NEWS_COLUMNS:
        if column in row.index:
            data[column] = row.get(column, "")

    for column in NEWS_DATE_COLUMNS:
        data[column] = to_yyyy_mm_dd(data.get(column, ""))

    if not str(data.get("first_seen_at", "") or "").strip():
        data["first_seen_at"] = applied_at

    return data


def _update_manual_decisions_applied_at(
    path: Path,
    applied_news_ids: set[str],
    applied_at: str,
) -> None:
    if not applied_news_ids:
        return

    decisions = read_csv_safe(path, MANUAL_DECISION_COLUMNS)

    if decisions.empty:
        return

    decisions["news_id"] = decisions["news_id"].fillna("").astype(str).str.strip()
    decisions["manual_decision"] = (
        decisions["manual_decision"].fillna("").astype(str).str.strip().str.lower()
    )
    original_dates = decisions[MANUAL_DECISION_DATE_COLUMNS].copy()

    for column in MANUAL_DECISION_DATE_COLUMNS:
        decisions[column] = decisions[column].map(to_yyyy_mm_dd)

    mask = (
        decisions["news_id"].isin(applied_news_ids)
        & decisions["manual_decision"].isin(["keep", "reject"])
        & (decisions["applied_at"].str.strip() == "")
    )

    if mask.any():
        decisions.loc[mask, "applied_at"] = applied_at

    if mask.any() or not decisions[MANUAL_DECISION_DATE_COLUMNS].equals(original_dates):
        atomic_write_csv(decisions, path, MANUAL_DECISION_COLUMNS)


def _apply_review_decisions() -> int:
    started_at = now_utc_iso()
    run_id = f"news_backfill_{compact_timestamp(started_at)}"
    manual_decisions = load_manual_decisions(MANUAL_DECISIONS_FILE)

    if manual_decisions.empty:
        print("No manual review decisions to apply.")
        return 0

    applied_at = utc_today_yyyy_mm_dd()
    keep_history = read_csv_safe(HISTORY_FILE, NEWS_COLUMNS)
    review = read_csv_safe(REVIEW_FILE, REVIEW_COLUMNS)
    reject_log = read_csv_safe(REJECT_FILE, REJECT_COLUMNS)
    rows_to_apply: list[dict[str, object]] = []
    missing_news_ids: list[str] = []

    for _, decision_row in manual_decisions.iterrows():
        news_id = str(decision_row.get("news_id", "") or "").strip()
        decision = str(decision_row.get("manual_decision", "") or "").strip().lower()
        source_row = None

        for frame in (keep_history, review, reject_log):
            source_row = _state_row_by_id(frame, news_id)

            if source_row is not None:
                break

        if source_row is None:
            missing_news_ids.append(news_id)
            continue

        row = _news_row_from_state(source_row, applied_at)
        row["rule_filter_status"] = str(
            row.get("rule_filter_status", "") or row.get("filter_status", "")
        )
        row["filter_status"] = decision
        row["filter_reason"] = f"manual_{decision}"
        row["manual_override"] = "True"
        row["last_seen_at"] = applied_at
        rows_to_apply.append(row)

    for news_id in missing_news_ids:
        print(
            "Warning: manual decision references unknown news_id; "
            f"will apply after future reprocess if seen: {news_id}"
        )

    if not rows_to_apply:
        print("No manual review decisions matched current state files.")
        return 0

    rows = pd.DataFrame(rows_to_apply, columns=NEWS_COLUMNS)
    _, review_df, reject = reconcile_news_statuses(
        rows=rows,
        history_path=HISTORY_FILE,
        review_path=REVIEW_FILE,
        reject_path=REJECT_FILE,
    )
    keep, _, _ = split_rows(rows)
    _update_manual_decisions_applied_at(
        MANUAL_DECISIONS_FILE,
        set(rows["news_id"].dropna().astype(str)),
        applied_at,
    )
    append_fetch_log(
        {
            "run_id": run_id,
            "started_at": started_at,
            "completed_at": now_utc_iso(),
            "source_id": SOURCE_ID,
            "mode": "apply_review_decisions",
            "window_start": "",
            "window_end": "",
            "pages_requested": 0,
            "items_fetched": len(rows),
            "keep_count": len(keep),
            "review_count": len(review_df),
            "reject_count": len(reject),
            "raw_file": "",
            "status": "success",
            "error_message": "",
        }
    )
    print(f"Applied {len(rows)} manual review decision(s).")
    return 0


def run_backfill(args: argparse.Namespace) -> int:
    if args.apply_review_decisions:
        try:
            return _apply_review_decisions()
        except Exception as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1

    started_at = now_utc_iso()
    run_id = f"news_backfill_{compact_timestamp(started_at)}"
    mode = "dry_run" if args.dry_run else "backfill"

    if args.reprocess_raw:
        mode = "reprocess_raw_dry_run" if args.dry_run else "reprocess_raw"

    pages_requested = 0
    raw_file = ""
    window_start = ""
    window_end = ""
    rows = pd.DataFrame(columns=NEWS_COLUMNS)
    keep = pd.DataFrame(columns=NEWS_COLUMNS)
    review = pd.DataFrame(columns=NEWS_COLUMNS)
    reject = pd.DataFrame(columns=REJECT_COLUMNS)

    try:
        rows, window_start, window_end, raw_file, pages_requested, _ = _fetch_and_classify(args)
        keep, review, reject = split_rows(rows)
        previous_frames = [
            read_csv_safe(CURRENT_NEWS_FILE, NEWS_COLUMNS),
            read_csv_safe(HISTORY_FILE, NEWS_COLUMNS),
            read_csv_safe(REVIEW_FILE, REVIEW_COLUMNS),
            read_csv_safe(REJECT_FILE, REJECT_COLUMNS),
        ]

        if not args.dry_run:
            latest_manifest = build_latest_fetch_manifest(
                rows=rows,
                previous_frames=previous_frames,
                fetch_run_at=started_at,
            )
            keep, review, reject = write_outputs(rows)
            write_latest_fetch_manifest(latest_manifest)

        if not args.dry_run:
            append_fetch_log(
                {
                    "run_id": run_id,
                    "started_at": started_at,
                    "completed_at": now_utc_iso(),
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
                    "error_message": "",
                }
            )

        print_summary(
            window_start=window_start,
            window_end=window_end,
            rows=rows,
            keep=keep,
            review=review,
            reject=reject,
            raw_file=raw_file if not args.dry_run else "",
        )
        return 0

    except Exception as error:
        error_message = str(error)

        if not args.dry_run:
            try:
                append_fetch_log(
                    {
                        "run_id": run_id,
                        "started_at": started_at,
                        "completed_at": now_utc_iso(),
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

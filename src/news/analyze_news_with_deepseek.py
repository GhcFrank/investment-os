"""
CLI for analyzing news review queue items with DeepSeek.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from news.deepseek_news_analyzer import (
    ANALYSIS_COLUMNS,
    COMPANY_UNIVERSE_TRUNCATED_MARKER,
    analyze_article_with_deepseek,
    load_company_universe,
    load_deepseek_config,
    load_deepseek_runtime_defaults,
    summarize_deepseek_error,
)


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_FILE = BASE_DIR / "data" / "news" / "news_review_queue.csv"
DEFAULT_OUTPUT_FILE = BASE_DIR / "data" / "news" / "deepseek_news_analysis.csv"
DEFAULT_LOG_FILE = BASE_DIR / "data" / "news" / "deepseek_news_analysis_log.csv"
DEFAULT_COMPANY_MASTER_FILE = BASE_DIR / "data" / "master" / "company_master.csv"

LOG_COLUMNS = [
    "run_at",
    "status",
    "articles_seen",
    "articles_analyzed",
    "articles_skipped",
    "articles_failed",
    "model",
    "prompt_version",
    "error_message",
]

INPUT_COLUMNS = [
    "news_id",
    "title",
    "url",
    "published_at_local",
    "published_at_gmt",
    "content_class",
    "source_quality",
    "matched_tickers",
    "matched_subthemes",
    "matched_keywords",
    "summary",
    "excerpt",
    "reason",
]

SELECTION_REASON_COLUMN = "_deepseek_selection_reason"


def now_utc_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_csv_safe(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)

    try:
        df = pd.read_csv(path, dtype=str)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)

    for column in columns:
        if column not in df.columns:
            df[column] = ""

    return df.reindex(columns=columns)


def write_csv(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = df.copy()

    for column in columns:
        if column not in output.columns:
            output[column] = ""

    temp_path = path.with_name(f".{path.name}.tmp")
    output.reindex(columns=columns).to_csv(temp_path, index=False, encoding="utf-8")
    temp_path.replace(path)


def append_log(path: Path, row: dict[str, object]) -> None:
    existing = read_csv_safe(path, LOG_COLUMNS)
    combined = pd.concat(
        [existing, pd.DataFrame([row], columns=LOG_COLUMNS)],
        ignore_index=True,
    )
    write_csv(combined, path, LOG_COLUMNS)


def load_input_articles(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    df = pd.read_csv(path, dtype=str)
    if "news_id" not in df.columns:
        raise ValueError("Input file must contain news_id column.")
    if "title" not in df.columns:
        raise ValueError("Input file must contain title column.")

    for column in INPUT_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    return df.reindex(columns=INPUT_COLUMNS)


def select_articles_to_analyze(
    articles: pd.DataFrame,
    existing: pd.DataFrame,
    model: str,
    prompt_version: str,
    limit: int,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    existing_statuses = _existing_status_by_news_id(existing, model, prompt_version)
    selected_rows = []
    skipped_rows = []

    for _, row in articles.iterrows():
        news_id = str(row.get("news_id", "") or "")
        existing_status = existing_statuses.get(news_id, "")

        if existing_status == "ok" and not force:
            skipped_rows.append(row.to_dict())
            continue

        reason = "new"
        if existing_status:
            reason = f"previous {existing_status}"

        item = row.to_dict()
        item[SELECTION_REASON_COLUMN] = reason
        selected_rows.append(item)

    selected = pd.DataFrame(selected_rows)
    skipped_ok = pd.DataFrame(skipped_rows)

    if selected.empty:
        selected = pd.DataFrame(columns=[*INPUT_COLUMNS, SELECTION_REASON_COLUMN])
    else:
        selected = selected.reindex(columns=[*INPUT_COLUMNS, SELECTION_REASON_COLUMN])

    if skipped_ok.empty:
        skipped_ok = pd.DataFrame(columns=INPUT_COLUMNS)
    else:
        skipped_ok = skipped_ok.reindex(columns=INPUT_COLUMNS)

    selected = selected.head(limit).copy()
    retry_failed = int(selected[SELECTION_REASON_COLUMN].eq("previous failed").sum())

    return selected, skipped_ok, retry_failed


def _existing_status_by_news_id(
    existing: pd.DataFrame,
    model: str,
    prompt_version: str,
) -> dict[str, str]:
    if existing.empty:
        return {}

    relevant = existing[
        (existing["model"].fillna("").astype(str) == model)
        & (existing["prompt_version"].fillna("").astype(str) == prompt_version)
    ].copy()
    if relevant.empty:
        return {}

    relevant = relevant.drop_duplicates(
        subset=["news_id", "model", "prompt_version"],
        keep="last",
    )
    return dict(
        zip(
            relevant["news_id"].fillna("").astype(str),
            relevant["status"].fillna("").astype(str).str.lower(),
            strict=False,
        )
    )


def upsert_analysis_rows(
    existing: pd.DataFrame,
    new_rows: pd.DataFrame,
) -> pd.DataFrame:
    combined = pd.concat([existing, new_rows], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=ANALYSIS_COLUMNS)

    combined = combined.drop_duplicates(
        subset=["news_id", "model", "prompt_version"],
        keep="last",
    ).reset_index(drop=True)
    return combined.reindex(columns=ANALYSIS_COLUMNS)


def run_analysis(args: argparse.Namespace) -> int:
    run_at = now_utc_iso()
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    log_file = Path(args.log_file)
    articles_analyzed = 0
    articles_failed = 0
    articles_skipped = 0
    retrying_previous_failed = 0

    try:
        defaults = load_deepseek_runtime_defaults()
        model = defaults["model"]
        prompt_version = defaults["prompt_version"]
        limit = args.limit if args.limit is not None else defaults["analysis_limit"]
        articles = load_input_articles(input_file)
        existing = read_csv_safe(output_file, ANALYSIS_COLUMNS)
        company_universe = load_company_universe(
            company_master_file=args.company_master_file,
            max_companies=args.max_company_universe,
        )
        company_universe_count = _company_universe_count(company_universe)
        selected, skipped_ok, retrying_previous_failed = select_articles_to_analyze(
            articles=articles,
            existing=existing,
            model=model,
            prompt_version=prompt_version,
            limit=limit,
            force=args.force,
        )
        articles_skipped = len(skipped_ok)

        if args.dry_run:
            print("Dry run only. No API calls will be made.")
            print(f"Loaded company universe: {company_universe_count} companies")
            print()
            print(f"To analyze: {len(selected)}")
            print(f"Skipped existing ok: {articles_skipped}")
            print(f"Retrying previous failed: {retrying_previous_failed}")
            print()
            print("To analyze:")
            for _, row in selected.iterrows():
                reason = row.get(SELECTION_REASON_COLUMN, "new")
                print(f"- {row['news_id']} | {reason} | {row['title']}")
            print()
            print("Skipped existing ok:")
            if skipped_ok.empty:
                print("- none")
            else:
                for _, row in skipped_ok.iterrows():
                    print(f"- {row['news_id']} | {row['title']}")
            return 0

        config = load_deepseek_config()
        model = config["model"]
        prompt_version = config["prompt_version"]
        selected, skipped_ok, retrying_previous_failed = select_articles_to_analyze(
            articles=articles,
            existing=existing,
            model=model,
            prompt_version=prompt_version,
            limit=limit,
            force=args.force,
        )
        articles_skipped = len(skipped_ok)
        rows = []

        for _, row in selected.iterrows():
            article = row.drop(labels=[SELECTION_REASON_COLUMN], errors="ignore").to_dict()
            result = analyze_article_with_deepseek(
                article=article,
                config=config,
                company_universe=company_universe,
            )
            rows.append(result)
            articles_analyzed += 1
            if result["status"] == "failed":
                articles_failed += 1
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

        if rows:
            new_rows = pd.DataFrame(rows, columns=ANALYSIS_COLUMNS)
            combined = upsert_analysis_rows(existing, new_rows)
            write_csv(combined, output_file, ANALYSIS_COLUMNS)
        elif not output_file.exists():
            write_csv(pd.DataFrame(columns=ANALYSIS_COLUMNS), output_file, ANALYSIS_COLUMNS)

        status = _log_status(
            articles_analyzed=articles_analyzed,
            articles_failed=articles_failed,
            articles_seen=len(articles),
            articles_skipped=articles_skipped,
        )
        append_log(
            log_file,
            {
                "run_at": run_at,
                "status": status,
                "articles_seen": len(articles),
                "articles_analyzed": articles_analyzed,
                "articles_skipped": articles_skipped,
                "articles_failed": articles_failed,
                "model": model,
                "prompt_version": prompt_version,
                "error_message": "",
            },
        )
        print(
            f"DeepSeek analysis completed: analyzed={articles_analyzed}, "
            f"skipped={articles_skipped}, failed={articles_failed}"
        )
        return 0
    except Exception as error:
        message = summarize_deepseek_error(error)
        if not args.dry_run:
            append_log(
                log_file,
                {
                    "run_at": run_at,
                    "status": "failed",
                    "articles_seen": 0,
                    "articles_analyzed": articles_analyzed,
                    "articles_skipped": articles_skipped,
                    "articles_failed": articles_failed,
                    "model": "",
                    "prompt_version": "",
                    "error_message": message,
                },
            )
        print(f"Error: {message}", file=sys.stderr)
        return 1


def _log_status(
    articles_analyzed: int,
    articles_failed: int,
    articles_seen: int,
    articles_skipped: int,
) -> str:
    if articles_analyzed == 0:
        return "skipped"
    if articles_failed == 0:
        return "ok"
    if articles_failed == articles_analyzed:
        return "failed"
    return "partial_failed"


def _company_universe_count(company_universe: list[str]) -> int:
    return sum(1 for item in company_universe if item != COMPANY_UNIVERSE_TRUNCATED_MARKER)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = load_deepseek_runtime_defaults()
    parser = argparse.ArgumentParser(description="Analyze news with DeepSeek.")
    parser.add_argument("--limit", type=int, default=defaults["analysis_limit"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--input-file", default=str(DEFAULT_INPUT_FILE))
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--log-file", default=str(DEFAULT_LOG_FILE))
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--company-master-file", default=str(DEFAULT_COMPANY_MASTER_FILE))
    parser.add_argument(
        "--max-company-universe",
        type=int,
        default=200,
        help=(
            "Maximum companies to include as DeepSeek prompt context. "
            "DEEPSEEK_MAX_INPUT_CHARS limits article context only; output "
            "schema and instructions are not truncated."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    sys.exit(run_analysis(parse_args()))


if __name__ == "__main__":
    main()

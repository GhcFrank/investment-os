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
    analyze_article_with_deepseek,
    load_deepseek_config,
    load_deepseek_runtime_defaults,
    summarize_deepseek_error,
)


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_FILE = BASE_DIR / "data" / "news" / "news_review_queue.csv"
DEFAULT_OUTPUT_FILE = BASE_DIR / "data" / "news" / "deepseek_news_analysis.csv"
DEFAULT_LOG_FILE = BASE_DIR / "data" / "news" / "deepseek_news_analysis_log.csv"

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
) -> tuple[pd.DataFrame, int]:
    if existing.empty or force:
        candidates = articles.copy()
        skipped = 0
    else:
        analyzed_keys = set(
            zip(
                existing["news_id"].fillna("").astype(str),
                existing["model"].fillna("").astype(str),
                existing["prompt_version"].fillna("").astype(str),
                strict=False,
            )
        )
        mask = ~articles["news_id"].fillna("").astype(str).map(
            lambda news_id: (news_id, model, prompt_version) in analyzed_keys
        )
        skipped = int((~mask).sum())
        candidates = articles[mask].copy()

    return candidates.head(limit), skipped


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
    )
    return combined.reindex(columns=ANALYSIS_COLUMNS)


def run_analysis(args: argparse.Namespace) -> int:
    run_at = now_utc_iso()
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    log_file = Path(args.log_file)
    articles_analyzed = 0
    articles_failed = 0
    articles_skipped = 0

    try:
        defaults = load_deepseek_runtime_defaults()
        model = defaults["model"]
        prompt_version = defaults["prompt_version"]
        limit = args.limit if args.limit is not None else defaults["analysis_limit"]
        articles = load_input_articles(input_file)
        existing = read_csv_safe(output_file, ANALYSIS_COLUMNS)
        selected, articles_skipped = select_articles_to_analyze(
            articles=articles,
            existing=existing,
            model=model,
            prompt_version=prompt_version,
            limit=limit,
            force=args.force,
        )

        if args.dry_run:
            print(f"Dry run: {len(selected)} article(s) would be analyzed.")
            for _, row in selected.iterrows():
                print(f"- {row['news_id']} | {row['title']}")
            return 0

        config = load_deepseek_config()
        model = config["model"]
        prompt_version = config["prompt_version"]
        selected, articles_skipped = select_articles_to_analyze(
            articles=articles,
            existing=existing,
            model=model,
            prompt_version=prompt_version,
            limit=limit,
            force=args.force,
        )
        rows = []

        for _, row in selected.iterrows():
            result = analyze_article_with_deepseek(row.to_dict(), config)
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

        status = _log_status(articles_analyzed, articles_failed)
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


def _log_status(articles_analyzed: int, articles_failed: int) -> str:
    if articles_analyzed == 0:
        return "ok"
    if articles_failed == 0:
        return "ok"
    if articles_failed == articles_analyzed:
        return "failed"
    return "partial_failed"


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
    return parser.parse_args(argv)


def main() -> None:
    sys.exit(run_analysis(parse_args()))


if __name__ == "__main__":
    main()

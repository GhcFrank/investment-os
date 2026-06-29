"""
CLI for manual DeepSeek Pro follow-up on Flash keep/watch news.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from news.analyze_news_with_deepseek import read_csv_safe, write_csv
from news.deepseek_news_analyzer import ANALYSIS_COLUMNS, summarize_deepseek_error
from news.deepseek_news_followup import (
    DEFAULT_ANALYSIS_FILE,
    DEFAULT_COMPANY_MASTER_FILE,
    DEFAULT_FOLLOWUP_FILE,
    DEFAULT_FOLLOWUP_LOG_FILE,
    DEFAULT_REVIEW_QUEUE_FILE,
    FOLLOWUP_COLUMNS,
    FOLLOWUP_LOG_COLUMNS,
    analyze_followup_with_deepseek,
    append_followup_log,
    load_deepseek_followup_config,
    load_flash_followup_candidates,
    load_company_universe,
    now_utc_iso,
    select_followup_candidates,
    upsert_followup_rows,
)
from news.news_filter import REVIEW_COLUMNS


def _decision_filter(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _log_status(reviewed: int, failed: int) -> str:
    if reviewed == 0:
        return "skipped"
    if failed == 0:
        return "ok"
    if failed == reviewed:
        return "failed"
    return "partial_failed"


def run_followup(args: argparse.Namespace) -> int:
    run_at = now_utc_iso()
    articles_reviewed = 0
    articles_failed = 0
    articles_skipped = 0

    analysis_file = Path(args.analysis_file)
    review_queue_file = Path(args.review_queue_file)
    output_file = Path(args.output_file)
    log_file = Path(args.log_file)

    try:
        config = load_deepseek_followup_config()
        analysis = read_csv_safe(analysis_file, ANALYSIS_COLUMNS)
        review_queue = read_csv_safe(review_queue_file, REVIEW_COLUMNS)
        existing = read_csv_safe(output_file, FOLLOWUP_COLUMNS)
        decisions = _decision_filter(args.decision_filter)
        candidates = load_flash_followup_candidates(
            analysis=analysis,
            review_queue=review_queue,
            source_model=args.source_model,
            source_prompt_version=args.source_prompt_version,
            decision_filter=decisions,
        )
        selected, articles_skipped = select_followup_candidates(
            candidates=candidates,
            existing=existing,
            followup_model=config["model"],
            followup_prompt_version=config["prompt_version"],
            limit=args.limit,
            retry_failed=args.retry_failed,
            force=args.force,
        )
        company_universe = load_company_universe(
            company_master_file=args.company_master_file,
            max_companies=args.max_company_universe,
        )

        rows = []
        for _, row in selected.iterrows():
            result = analyze_followup_with_deepseek(
                candidate=row.to_dict(),
                config=config,
                company_universe=company_universe,
            )
            rows.append(result)
            articles_reviewed += 1
            if result["status"] == "failed":
                articles_failed += 1
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

        if rows:
            new_rows = pd.DataFrame(rows, columns=FOLLOWUP_COLUMNS)
            combined = upsert_followup_rows(existing, new_rows)
            write_csv(combined, output_file, FOLLOWUP_COLUMNS)
        elif not output_file.exists():
            write_csv(pd.DataFrame(columns=FOLLOWUP_COLUMNS), output_file, FOLLOWUP_COLUMNS)

        append_followup_log(
            log_file,
            {
                "run_at": run_at,
                "status": _log_status(articles_reviewed, articles_failed),
                "articles_seen": len(candidates),
                "articles_reviewed": articles_reviewed,
                "articles_skipped": articles_skipped,
                "articles_failed": articles_failed,
                "source_model": args.source_model,
                "source_prompt_version": args.source_prompt_version,
                "followup_model": config["model"],
                "followup_prompt_version": config["prompt_version"],
                "error_message": "",
            },
        )
        print(
            "DeepSeek follow-up completed: "
            f"reviewed={articles_reviewed}, skipped={articles_skipped}, "
            f"failed={articles_failed}"
        )
        return 0
    except Exception as error:
        message = summarize_deepseek_error(error)
        try:
            append_followup_log(
                log_file,
                {
                    "run_at": run_at,
                    "status": "failed",
                    "articles_seen": 0,
                    "articles_reviewed": articles_reviewed,
                    "articles_skipped": articles_skipped,
                    "articles_failed": articles_failed,
                    "source_model": args.source_model,
                    "source_prompt_version": args.source_prompt_version,
                    "followup_model": "",
                    "followup_prompt_version": "",
                    "error_message": message,
                },
            )
        except Exception:
            pass
        print(f"Error: {message}", file=sys.stderr)
        return 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review Flash keep/watch news with DeepSeek Pro.",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--source-model", default="deepseek-v4-flash")
    parser.add_argument(
        "--source-prompt-version",
        default="deepseek_flash_preprocess_v1",
    )
    parser.add_argument("--decision-filter", default="keep,watch")
    parser.add_argument("--analysis-file", default=str(DEFAULT_ANALYSIS_FILE))
    parser.add_argument("--review-queue-file", default=str(DEFAULT_REVIEW_QUEUE_FILE))
    parser.add_argument("--output-file", default=str(DEFAULT_FOLLOWUP_FILE))
    parser.add_argument("--log-file", default=str(DEFAULT_FOLLOWUP_LOG_FILE))
    parser.add_argument("--company-master-file", default=str(DEFAULT_COMPANY_MASTER_FILE))
    parser.add_argument("--max-company-universe", type=int, default=200)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    sys.exit(run_followup(parse_args()))


if __name__ == "__main__":
    main()


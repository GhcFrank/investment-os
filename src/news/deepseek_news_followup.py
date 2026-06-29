"""
DeepSeek Pro follow-up helpers for Flash keep/watch news.
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from news.analyze_news_with_deepseek import read_csv_safe, write_csv
from news.deepseek_news_analyzer import (
    BASE_DIR,
    ENV_FILE,
    load_company_universe,
    load_deepseek_config,
    parse_deepseek_json_response,
    summarize_deepseek_error,
    is_retryable_deepseek_error,
    _thinking_kwargs,
)


NEWS_DIR = BASE_DIR / "data" / "news"
DEFAULT_ANALYSIS_FILE = NEWS_DIR / "deepseek_news_analysis.csv"
DEFAULT_REVIEW_QUEUE_FILE = NEWS_DIR / "news_review_queue.csv"
DEFAULT_FOLLOWUP_FILE = NEWS_DIR / "deepseek_news_followup.csv"
DEFAULT_FOLLOWUP_LOG_FILE = NEWS_DIR / "deepseek_news_followup_log.csv"
DEFAULT_COMPANY_MASTER_FILE = BASE_DIR / "data" / "master" / "company_master.csv"

FOLLOWUP_COLUMNS = [
    "news_id",
    "followup_date",
    "source_model",
    "source_prompt_version",
    "followup_model",
    "followup_prompt_version",
    "status",
    "error_message",
    "research_signal",
    "direct_tickers",
    "second_order_tickers",
    "affected_subthemes",
    "investment_thesis",
    "why_it_matters",
    "what_to_verify",
    "data_to_check",
    "keywords_to_track",
    "customer_supplier_links",
    "earnings_call_questions",
    "confidence",
    "input_chars",
    "output_chars",
    "input_tokens",
    "output_tokens",
]

FOLLOWUP_LOG_COLUMNS = [
    "run_at",
    "status",
    "articles_seen",
    "articles_reviewed",
    "articles_skipped",
    "articles_failed",
    "source_model",
    "source_prompt_version",
    "followup_model",
    "followup_prompt_version",
    "error_message",
]


def utc_today() -> str:
    return datetime.now(UTC).date().isoformat()


def now_utc_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_deepseek_followup_config() -> dict:
    load_dotenv(ENV_FILE)
    config = load_deepseek_config()
    config["model"] = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro").strip()
    config["prompt_version"] = os.getenv(
        "DEEPSEEK_PROMPT_VERSION",
        "deepseek_pro_followup_v1",
    ).strip()
    config["thinking"] = os.getenv("DEEPSEEK_THINKING", "disabled").strip()
    return config


def load_flash_followup_candidates(
    analysis: pd.DataFrame,
    review_queue: pd.DataFrame,
    source_model: str = "deepseek-v4-flash",
    source_prompt_version: str = "deepseek_flash_preprocess_v1",
    decision_filter: list[str] | None = None,
) -> pd.DataFrame:
    decisions = {
        decision.strip().lower()
        for decision in (decision_filter or ["keep", "watch"])
        if decision.strip()
    }

    if analysis.empty:
        return pd.DataFrame()

    work = analysis.copy()
    for column in [
        "news_id",
        "model",
        "prompt_version",
        "status",
        "recommended_decision",
    ]:
        if column not in work.columns:
            work[column] = ""

    mask = (
        (work["model"].fillna("").astype(str) == source_model)
        & (work["prompt_version"].fillna("").astype(str) == source_prompt_version)
        & (work["status"].fillna("").astype(str).str.lower() == "ok")
        & (work["recommended_decision"].fillna("").astype(str).str.lower().isin(decisions))
    )
    work = work[mask].copy()
    if work.empty:
        return work

    if not review_queue.empty and "news_id" in review_queue.columns:
        article_cols = [
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
            "excerpt",
        ]
        queue = review_queue.copy()
        for column in article_cols:
            if column not in queue.columns:
                queue[column] = ""
        queue = queue.drop_duplicates(subset=["news_id"], keep="last")
        work = work.merge(
            queue[article_cols],
            on="news_id",
            how="left",
            suffixes=("", "_article"),
        )

    return work.reset_index(drop=True)


def select_followup_candidates(
    candidates: pd.DataFrame,
    existing: pd.DataFrame,
    followup_model: str,
    followup_prompt_version: str,
    limit: int,
    retry_failed: bool = False,
    force: bool = False,
) -> tuple[pd.DataFrame, int]:
    if candidates.empty:
        return candidates.copy(), 0

    existing_statuses = _existing_followup_statuses(
        existing,
        followup_model,
        followup_prompt_version,
    )
    selected_rows = []
    skipped = 0

    for _, row in candidates.iterrows():
        news_id = str(row.get("news_id", "") or "")
        status = existing_statuses.get(news_id, "")

        if status and not force:
            if retry_failed and status == "failed":
                pass
            else:
                skipped += 1
                continue

        selected_rows.append(row.to_dict())

    selected = pd.DataFrame(selected_rows)
    if selected.empty:
        selected = pd.DataFrame(columns=candidates.columns)

    return selected.head(limit).copy(), skipped


def _existing_followup_statuses(
    existing: pd.DataFrame,
    followup_model: str,
    followup_prompt_version: str,
) -> dict[str, str]:
    if existing.empty:
        return {}

    for column in ["news_id", "followup_model", "followup_prompt_version", "status"]:
        if column not in existing.columns:
            existing[column] = ""

    relevant = existing[
        (existing["followup_model"].fillna("").astype(str) == followup_model)
        & (
            existing["followup_prompt_version"].fillna("").astype(str)
            == followup_prompt_version
        )
    ].copy()
    if relevant.empty:
        return {}

    relevant = relevant.drop_duplicates(
        subset=["news_id", "followup_model", "followup_prompt_version"],
        keep="last",
    )
    return dict(
        zip(
            relevant["news_id"].fillna("").astype(str),
            relevant["status"].fillna("").astype(str).str.lower(),
            strict=False,
        )
    )


def build_followup_prompt(
    candidate: dict,
    company_universe: list[str] | None = None,
    max_input_chars: int = 6000,
) -> list[dict]:
    system = (
        "You are an investment research analyst. This is not first-pass "
        "keep/watch/reject classification. The Flash model already marked this "
        "article keep or watch. Extract further investment research leads. "
        "Return JSON only. Output a single valid JSON object. Do not include "
        "Markdown or commentary outside the JSON object."
    )
    schema = {
        "research_signal": "Core research signal.",
        "direct_tickers": ["ANET"],
        "second_order_tickers": ["MRVL"],
        "affected_subthemes": ["Networking"],
        "investment_thesis": "Short thesis.",
        "why_it_matters": "Why it matters.",
        "what_to_verify": ["Question or claim to verify"],
        "data_to_check": ["Dataset or metric"],
        "keywords_to_track": ["keyword"],
        "customer_supplier_links": ["relationship"],
        "earnings_call_questions": ["question"],
        "confidence": "integer 1-5",
    }
    article_context = _truncate(
        "\n".join(
            [
                f"news_id: {_clean(candidate.get('news_id'))}",
                f"title: {_clean(candidate.get('title'))}",
                f"url: {_clean(candidate.get('url'))}",
                f"published_at_gmt: {_clean(candidate.get('published_at_gmt'))}",
                f"matched_tickers: {_clean(candidate.get('matched_tickers'))}",
                f"matched_subthemes: {_clean(candidate.get('matched_subthemes'))}",
                f"matched_keywords: {_clean(candidate.get('matched_keywords'))}",
                f"excerpt: {_clean(candidate.get('excerpt'))}",
                f"flash_recommended_decision: {_clean(candidate.get('recommended_decision'))}",
                f"flash_summary: {_clean(candidate.get('summary'))}",
                f"flash_why_it_matters: {_clean(candidate.get('why_it_matters'))}",
            ]
        ),
        max_input_chars,
    )
    universe = "\n".join(company_universe or ["(none loaded)"])
    user = (
        "Supported company universe:\n"
        + universe
        + "\n\nArticle and Flash context:\n"
        + article_context
        + "\n\nOutput requirements:\n"
        "Return JSON only.\n"
        "Output a single valid JSON object.\n"
        "Do not wrap the JSON in Markdown.\n"
        "Include direct tickers, second-order tickers, affected subthemes, "
        "investment thesis, data to verify, tracking keywords, customer/supplier "
        "links, earnings call questions, and confidence.\n"
        "JSON schema:\n"
        + json.dumps(schema, ensure_ascii=False)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def analyze_followup_with_deepseek(
    candidate: dict,
    config: dict,
    company_universe: list[str] | None = None,
) -> dict:
    max_input_chars = int(config.get("max_input_chars", 6000) or 6000)
    messages = build_followup_prompt(candidate, company_universe, max_input_chars)
    input_chars = sum(len(str(message.get("content", ""))) for message in messages)
    max_retries = max(0, int(config.get("max_retries", 2) or 0))
    retry_sleep_seconds = max(0.0, float(config.get("retry_sleep_seconds", 2.0) or 0.0))
    total_attempts = max_retries + 1
    last_error = "DeepSeek API request failed"

    for attempt in range(total_attempts):
        try:
            client = OpenAI(
                api_key=config["api_key"],
                base_url=config.get("base_url", "https://api.deepseek.com"),
            )
            response = client.chat.completions.create(
                model=config.get("model", "deepseek-v4-pro"),
                messages=messages,
                max_tokens=int(config.get("max_output_tokens", 1500) or 1500),
                temperature=float(config.get("temperature", 0.1) or 0.1),
                response_format={"type": "json_object"},
                **_thinking_kwargs(config),
            )
            content = _extract_response_content(response)
            parsed = parse_deepseek_json_response(content)
            usage = getattr(response, "usage", None)
            return normalize_followup_result(
                candidate=candidate,
                raw_result=parsed,
                config=config,
                status="ok",
                input_chars=input_chars,
                output_chars=len(content),
                input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
                output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            )
        except Exception as error:
            short_error = summarize_deepseek_error(error)
            last_error = short_error
            retryable = is_retryable_deepseek_error(
                short_error
            ) or is_retryable_deepseek_error(error)
            if attempt < max_retries and retryable:
                print(
                    f"DeepSeek follow-up failed for {_clean(candidate.get('news_id'))} "
                    f"on attempt {attempt + 1}/{total_attempts}: {short_error}. "
                    f"Retrying in {retry_sleep_seconds}s..."
                )
                if retry_sleep_seconds > 0:
                    time.sleep(retry_sleep_seconds)
                continue

            return normalize_followup_result(
                candidate=candidate,
                raw_result={},
                config=config,
                status="failed",
                error_message=last_error,
                input_chars=input_chars,
                output_chars=0,
            )

    return normalize_followup_result(
        candidate=candidate,
        raw_result={},
        config=config,
        status="failed",
        error_message=last_error,
        input_chars=input_chars,
        output_chars=0,
    )


def normalize_followup_result(
    candidate: dict,
    raw_result: dict,
    config: dict,
    status: str = "ok",
    error_message: str = "",
    input_chars: int | None = None,
    output_chars: int | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> dict:
    result = {
        "news_id": _clean(candidate.get("news_id")),
        "followup_date": utc_today(),
        "source_model": _clean(candidate.get("model") or candidate.get("source_model")),
        "source_prompt_version": _clean(
            candidate.get("prompt_version") or candidate.get("source_prompt_version")
        ),
        "followup_model": _clean(config.get("model")),
        "followup_prompt_version": _clean(config.get("prompt_version")),
        "status": status,
        "error_message": error_message,
        "research_signal": _truncate(_clean(raw_result.get("research_signal")), 500),
        "direct_tickers": _join_list(raw_result.get("direct_tickers")),
        "second_order_tickers": _join_list(raw_result.get("second_order_tickers")),
        "affected_subthemes": _join_list(raw_result.get("affected_subthemes")),
        "investment_thesis": _truncate(_clean(raw_result.get("investment_thesis")), 1000),
        "why_it_matters": _truncate(_clean(raw_result.get("why_it_matters")), 1000),
        "what_to_verify": _join_list(raw_result.get("what_to_verify"), separator=" | "),
        "data_to_check": _join_list(raw_result.get("data_to_check"), separator=" | "),
        "keywords_to_track": _join_list(raw_result.get("keywords_to_track")),
        "customer_supplier_links": _join_list(
            raw_result.get("customer_supplier_links"),
            separator=" | ",
        ),
        "earnings_call_questions": _join_list(
            raw_result.get("earnings_call_questions"),
            separator=" | ",
        ),
        "confidence": _score(raw_result.get("confidence")),
        "input_chars": "" if input_chars is None else str(input_chars),
        "output_chars": "" if output_chars is None else str(output_chars),
        "input_tokens": "" if input_tokens is None else str(input_tokens),
        "output_tokens": "" if output_tokens is None else str(output_tokens),
    }
    return {column: result.get(column, "") for column in FOLLOWUP_COLUMNS}


def upsert_followup_rows(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([existing, new_rows], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=FOLLOWUP_COLUMNS)
    combined = combined.drop_duplicates(
        subset=["news_id", "followup_model", "followup_prompt_version"],
        keep="last",
    ).reset_index(drop=True)
    return combined.reindex(columns=FOLLOWUP_COLUMNS)


def append_followup_log(path: Path, row: dict[str, object]) -> None:
    existing = read_csv_safe(path, FOLLOWUP_LOG_COLUMNS)
    combined = pd.concat(
        [existing, pd.DataFrame([row], columns=FOLLOWUP_LOG_COLUMNS)],
        ignore_index=True,
    )
    write_csv(combined, path, FOLLOWUP_LOG_COLUMNS)


def _extract_response_content(response: Any) -> str:
    choices = getattr(response, "choices", None) if response is not None else None
    if not choices:
        raise ValueError("empty_response")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if content is None or not str(content).strip():
        raise ValueError("empty_response")
    return str(content).strip()


def _clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _truncate(value: str, max_length: int) -> str:
    if len(value) > max_length:
        return value[:max_length] + "\n[CONTEXT_TRUNCATED]"
    return value


def _join_list(value: object, separator: str = ",") -> str:
    if value is None:
        items: list[str] = []
    elif isinstance(value, list):
        items = [_clean(item) for item in value]
    elif isinstance(value, str):
        if "|" in value:
            items = [_clean(item) for item in value.split("|")]
        elif "," in value:
            items = [_clean(item) for item in value.split(",")]
        else:
            items = [_clean(value)] if value.strip() else []
    else:
        items = [_clean(value)]
    return separator.join(item for item in items if item)


def _score(value: object) -> str:
    try:
        numeric = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return ""
    return str(max(1, min(5, numeric)))


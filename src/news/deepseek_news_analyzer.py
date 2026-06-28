"""
Core helpers for DeepSeek-powered news analysis.
"""

from __future__ import annotations

import json
import os
import re
import time
import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BASE_DIR / ".env"
DEFAULT_COMPANY_MASTER_FILE = Path("data/master/company_master.csv")
COMPANY_UNIVERSE_TRUNCATED_MARKER = "[COMPANY_UNIVERSE_TRUNCATED]"

ANALYSIS_COLUMNS = [
    "news_id",
    "analysis_date",
    "prompt_version",
    "model",
    "status",
    "error_message",
    "relevance_score",
    "impact_score",
    "impact_direction",
    "signal_type",
    "recommended_decision",
    "primary_tickers",
    "secondary_tickers",
    "primary_subthemes",
    "summary",
    "why_it_matters",
    "follow_up_questions",
    "confidence",
    "input_chars",
    "output_chars",
    "input_tokens",
    "output_tokens",
]

VALID_IMPACT_DIRECTIONS = {"positive", "negative", "mixed", "neutral", "unclear"}
VALID_SIGNAL_TYPES = {
    "demand",
    "supply",
    "capex",
    "product",
    "competition",
    "customer",
    "regulation",
    "macro",
    "earnings",
    "financing",
    "noise",
}
VALID_DECISIONS = {"keep", "reject", "watch"}


def utc_today() -> str:
    return datetime.now(UTC).date().isoformat()


def load_deepseek_config() -> dict:
    """
    Read DeepSeek config from environment variables.
    """

    load_dotenv(ENV_FILE)
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Please configure it in .env or environment variables."
        )

    return {
        "api_key": api_key,
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip(),
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip(),
        "prompt_version": os.getenv(
            "DEEPSEEK_PROMPT_VERSION",
            "deepseek_news_v1",
        ).strip(),
        "analysis_limit": _env_int("DEEPSEEK_NEWS_ANALYSIS_LIMIT", 20),
        "max_input_chars": _env_int("DEEPSEEK_MAX_INPUT_CHARS", 6000),
        "max_output_tokens": _env_int("DEEPSEEK_MAX_OUTPUT_TOKENS", 1500),
        "temperature": _env_float("DEEPSEEK_TEMPERATURE", 0.1),
        "max_retries": _env_int("DEEPSEEK_MAX_RETRIES", 2, minimum=0),
        "retry_sleep_seconds": _env_float(
            "DEEPSEEK_RETRY_SLEEP_SECONDS",
            2.0,
            minimum=0.0,
        ),
    }


def load_deepseek_runtime_defaults() -> dict:
    """
    Load non-secret defaults for dry-run planning.
    """

    load_dotenv(ENV_FILE)
    return {
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip(),
        "prompt_version": os.getenv(
            "DEEPSEEK_PROMPT_VERSION",
            "deepseek_news_v1",
        ).strip(),
        "analysis_limit": _env_int("DEEPSEEK_NEWS_ANALYSIS_LIMIT", 20),
        "max_input_chars": _env_int("DEEPSEEK_MAX_INPUT_CHARS", 6000),
        "max_output_tokens": _env_int("DEEPSEEK_MAX_OUTPUT_TOKENS", 1500),
        "temperature": _env_float("DEEPSEEK_TEMPERATURE", 0.1),
        "max_retries": _env_int("DEEPSEEK_MAX_RETRIES", 2, minimum=0),
        "retry_sleep_seconds": _env_float(
            "DEEPSEEK_RETRY_SLEEP_SECONDS",
            2.0,
            minimum=0.0,
        ),
    }


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    if minimum is not None and value < minimum:
        return default
    return value


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    if minimum is not None and value < minimum:
        return default
    return value


def load_company_universe(
    company_master_file: Path | str = DEFAULT_COMPANY_MASTER_FILE,
    max_companies: int | None = None,
) -> list[str]:
    """
    Load company universe from company_master.csv for DeepSeek prompt context.

    This function is read-only and returns compact rows like:
    "ANET | Arista Networks | AI Infrastructure | Networking | Connectivity"
    """

    path = Path(company_master_file)
    if not path.is_absolute():
        path = BASE_DIR / path
    if not path.exists():
        return []

    limit = None
    if max_companies is not None:
        limit = max(0, int(max_companies))

    universe = []
    seen = set()
    truncated = False

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            ticker = _clean_text(row.get("ticker", "")).upper()
            if not ticker:
                continue
            item = " | ".join(
                [
                    ticker,
                    _clean_text(row.get("company", "")),
                    _clean_text(row.get("theme", "")),
                    _clean_text(row.get("subtheme", "")),
                    _clean_text(row.get("supply_chain_layer", "")),
                ]
            )
            if item in seen:
                continue
            if limit is not None and len(universe) >= limit:
                truncated = True
                break
            seen.add(item)
            universe.append(item)

    if truncated:
        universe.append(COMPANY_UNIVERSE_TRUNCATED_MARKER)

    return universe


def build_news_analysis_prompt(
    article: dict,
    company_universe: list[str] | None = None,
    max_input_chars: int = 6000,
) -> list[dict]:
    """
    Build OpenAI-compatible chat messages for DeepSeek.
    """

    system = (
        "You are an investment research assistant for an AI infrastructure and "
        "software watchlist. Your job is to classify and summarize news articles. "
        "Return JSON only. Do not include Markdown. Do not include explanations "
        "outside JSON. Use the supported company universe to identify directly "
        "and indirectly related tickers. Do not invent tickers outside the "
        "provided universe unless the article explicitly names a public company "
        "ticker. If the article is irrelevant, use "
        'recommended_decision="reject" and signal_type="noise".'
    )
    schema = {
        "relevance_score": "integer 1-5",
        "impact_score": "integer 1-5",
        "impact_direction": "positive | negative | mixed | neutral | unclear",
        "signal_type": (
            "demand | supply | capex | product | competition | customer | "
            "regulation | macro | earnings | financing | noise"
        ),
        "recommended_decision": "keep | reject | watch",
        "primary_tickers": ["ANET", "MRVL"],
        "secondary_tickers": ["COHR", "LITE"],
        "primary_subthemes": ["Networking", "Optical"],
        "summary": "One sentence summary, <= 300 chars.",
        "why_it_matters": "Why this matters to the watchlist, <= 500 chars.",
        "follow_up_questions": ["Question 1", "Question 2"],
        "confidence": "integer 1-5",
    }

    company_universe_block = _build_company_universe_block(company_universe or [])
    article_context_block = build_article_context(article, max_input_chars)
    output_schema_block = (
        "Return JSON only. Do not include Markdown or commentary.\n"
        "Output a single valid JSON object.\n"
        "Do not wrap the JSON in Markdown.\n"
        "Do not include commentary outside the JSON object.\n"
        "Allowed impact_direction values: positive, negative, mixed, neutral, unclear.\n"
        "Allowed signal_type values: demand, supply, capex, product, competition, "
        "customer, regulation, macro, earnings, financing, noise.\n"
        "Allowed recommended_decision values: keep, reject, watch.\n"
        "JSON schema:\n"
        + json.dumps(schema, ensure_ascii=False)
    )
    user = (
        "Analyze this news article for the AI infrastructure and software watchlist.\n\n"
        + company_universe_block
        + "\n\nCompany universe rules:\n"
        "Use the supported company universe to identify directly and indirectly related tickers.\n"
        "Do not invent tickers outside the provided universe unless the article explicitly names a public company ticker.\n"
        "If no supported ticker is directly relevant, leave primary_tickers empty and use secondary_tickers only when there is a clear supply-chain connection.\n"
        "Do not force ticker associations just because a company appears in the universe.\n\n"
        "Article context:\n"
        + article_context_block
        + "\n\nOutput requirements:\n"
        + output_schema_block
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _build_company_universe_block(company_universe: list[str]) -> str:
    lines = ["Supported company universe:"]
    if company_universe:
        lines.extend(company_universe)
    else:
        lines.append("(none loaded)")
    return "\n".join(lines)


def build_article_context(article: dict, max_chars: int) -> str:
    """
    Build article-only context and truncate only this section.
    """

    fields = [
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
    context = "\n".join(
        f"{field}: {_clean_text(article.get(field, ''))}" for field in fields
    )
    max_chars = max(0, int(max_chars or 0))
    if len(context) > max_chars:
        return context[:max_chars] + "\n[ARTICLE_CONTEXT_TRUNCATED]"
    return context


def parse_deepseek_json_response(content: str) -> dict:
    """
    Parse model response into a dict.
    """

    text = str(content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("invalid_json_response") from None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            raise ValueError("invalid_json_response") from None

    if not isinstance(parsed, dict):
        raise ValueError("invalid_json_response")

    return parsed


def analyze_article_with_deepseek(
    article: dict,
    config: dict,
    company_universe: list[str] | None = None,
) -> dict:
    """
    Call DeepSeek API for one article and return a normalized analysis row.
    """

    max_input_chars = int(config.get("max_input_chars", 6000) or 6000)
    messages = build_news_analysis_prompt(
        article=article,
        company_universe=company_universe,
        max_input_chars=max_input_chars,
    )
    input_chars = sum(len(str(message.get("content", ""))) for message in messages)

    max_retries = int(config.get("max_retries", 2) or 0)
    max_retries = max(0, max_retries)
    retry_sleep_seconds = float(config.get("retry_sleep_seconds", 2.0) or 0.0)
    retry_sleep_seconds = max(0.0, retry_sleep_seconds)
    total_attempts = max_retries + 1
    last_error = "DeepSeek API request failed"

    for attempt in range(total_attempts):
        try:
            client = OpenAI(
                api_key=config["api_key"],
                base_url=config.get("base_url", "https://api.deepseek.com"),
            )
            response = client.chat.completions.create(
                model=config.get("model", "deepseek-v4-flash"),
                messages=messages,
                max_tokens=int(config.get("max_output_tokens", 1500) or 1500),
                temperature=float(config.get("temperature", 0.1) or 0.1),
                response_format={"type": "json_object"},
            )
            content = _extract_response_content(response)
            parsed = parse_deepseek_json_response(content)
            usage = getattr(response, "usage", None)
            return normalize_analysis_result(
                article=article,
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
                news_id = _clean_text(article.get("news_id", ""))
                print(
                    f"DeepSeek analysis failed for {news_id} on attempt "
                    f"{attempt + 1}/{total_attempts}: {short_error}. "
                    f"Retrying in {retry_sleep_seconds}s..."
                )
                if retry_sleep_seconds > 0:
                    time.sleep(retry_sleep_seconds)
                continue

            return normalize_analysis_result(
                article=article,
                raw_result={},
                config=config,
                status="failed",
                error_message=last_error,
                input_chars=input_chars,
                output_chars=0,
            )

    return normalize_analysis_result(
        article=article,
        raw_result={},
        config=config,
        status="failed",
        error_message=last_error,
        input_chars=input_chars,
        output_chars=0,
    )


def _extract_response_content(response: Any) -> str:
    if response is None:
        raise ValueError("empty_response")

    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("empty_response")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        raise ValueError("empty_response")

    content = getattr(message, "content", None)
    if content is None or not str(content).strip():
        raise ValueError("empty_response")

    return str(content).strip()


def is_retryable_deepseek_error(error: Exception | str) -> bool:
    """
    Return True if a DeepSeek article analysis failure is likely transient.
    """

    text = str(error).lower()
    non_retryable_terms = [
        "401",
        "authentication failed",
        "402",
        "balance is insufficient",
        "insufficient balance",
        "missing deepseek_api_key",
        "deepseek_api_key is not set",
        "invalid request",
        "bad input",
    ]
    if any(term in text for term in non_retryable_terms):
        return False

    retryable_terms = [
        "empty_response",
        "invalid_json_response",
        "timeout",
        "timed out",
        "429",
        "rate limit",
        "temporarily unavailable",
        "connection error",
        "server error",
        "500",
        "502",
        "503",
        "504",
    ]
    return any(term in text for term in retryable_terms)


def normalize_analysis_result(
    article: dict,
    raw_result: dict,
    config: dict,
    status: str = "ok",
    error_message: str = "",
    input_chars: int | None = None,
    output_chars: int | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> dict:
    """
    Convert parsed model JSON into fixed CSV output schema.
    """

    result = {
        "news_id": _clean_text(article.get("news_id", "")),
        "analysis_date": utc_today(),
        "prompt_version": _clean_text(config.get("prompt_version", "")),
        "model": _clean_text(config.get("model", "")),
        "status": status,
        "error_message": error_message,
        "relevance_score": _score(raw_result.get("relevance_score")),
        "impact_score": _score(raw_result.get("impact_score")),
        "impact_direction": _choice(
            raw_result.get("impact_direction"),
            VALID_IMPACT_DIRECTIONS,
            "unclear",
        ),
        "signal_type": _choice(raw_result.get("signal_type"), VALID_SIGNAL_TYPES, "noise"),
        "recommended_decision": _choice(
            raw_result.get("recommended_decision"),
            VALID_DECISIONS,
            "watch",
        ),
        "primary_tickers": _join_list(raw_result.get("primary_tickers")),
        "secondary_tickers": _join_list(raw_result.get("secondary_tickers")),
        "primary_subthemes": _join_list(raw_result.get("primary_subthemes")),
        "summary": _truncate(_clean_text(raw_result.get("summary", "")), 300),
        "why_it_matters": _truncate(
            _clean_text(raw_result.get("why_it_matters", "")),
            500,
        ),
        "follow_up_questions": _join_list(
            raw_result.get("follow_up_questions"),
            separator=" | ",
            max_items=5,
        ),
        "confidence": _score(raw_result.get("confidence")),
        "input_chars": "" if input_chars is None else str(input_chars),
        "output_chars": "" if output_chars is None else str(output_chars),
        "input_tokens": "" if input_tokens is None else str(input_tokens),
        "output_tokens": "" if output_tokens is None else str(output_tokens),
    }

    return {column: result.get(column, "") for column in ANALYSIS_COLUMNS}


def summarize_deepseek_error(error: Exception | str) -> str:
    """
    Return a short user-readable error message.
    """

    text = str(error)
    lowered = text.lower()
    if "401" in lowered:
        return "DeepSeek API authentication failed"
    if "402" in lowered or "insufficient balance" in lowered:
        return "DeepSeek API balance is insufficient"
    if "429" in lowered or "rate limit" in lowered:
        return "DeepSeek API rate limit exceeded"
    if "timeout" in lowered or "timed out" in lowered:
        return "DeepSeek API request timeout"
    if "invalid_json_response" in lowered:
        return "invalid_json_response"
    if "empty_response" in lowered:
        return "empty_response"

    cleaned = re.sub(r"\s+", " ", text).strip()
    return _truncate(cleaned or "DeepSeek API request failed", 200)


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _truncate(value: str, max_length: int) -> str:
    return value[:max_length]


def _score(value: object) -> str:
    try:
        numeric = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return ""
    return str(max(1, min(5, numeric)))


def _choice(value: object, allowed: set[str], default: str) -> str:
    text = _clean_text(value).lower()
    return text if text in allowed else default


def _join_list(
    value: object,
    separator: str = ",",
    max_items: int | None = None,
) -> str:
    if value is None:
        values: list[str] = []
    elif isinstance(value, list):
        values = [_clean_text(item) for item in value]
    elif isinstance(value, str):
        if "|" in value:
            values = [_clean_text(item) for item in value.split("|")]
        elif "," in value:
            values = [_clean_text(item) for item in value.split(",")]
        else:
            values = [_clean_text(value)] if value.strip() else []
    else:
        values = [_clean_text(value)]

    values = [item for item in values if item]
    if max_items is not None:
        values = values[:max_items]

    return separator.join(values)

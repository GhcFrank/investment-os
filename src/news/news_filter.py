"""
Rule-based News Discovery V1 filtering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from news.news_utils import (
    atomic_write_csv,
    clean_html_text,
    contains_normalized_phrase,
    contains_phrase,
    csv_bool,
    join_values,
    normalize_match_text,
    normalize_url,
    normalize_wp_gmt_datetime,
    read_csv_safe,
    safe_int,
)


BASE_DIR = Path(__file__).resolve().parents[2]
MASTER_DIR = BASE_DIR / "data" / "master"
MANUAL_DECISIONS_FILE = BASE_DIR / "data" / "news" / "news_manual_decisions.csv"

FILTER_RULE_VERSION = "news_filter_v1_1"
ALIAS_CONTEXT_WINDOW_TOKENS = 12

NEWS_COLUMNS = [
    "news_id",
    "source_id",
    "source_post_id",
    "published_at_local",
    "published_at_gmt",
    "modified_at_gmt",
    "title",
    "excerpt",
    "url",
    "author_id",
    "category_ids",
    "category_names",
    "tag_ids",
    "tag_names",
    "content_class",
    "source_quality",
    "matched_tickers",
    "matched_aliases",
    "matched_themes",
    "matched_subthemes",
    "matched_keywords",
    "matched_exclusions",
    "company_score",
    "theme_score",
    "taxonomy_score",
    "exclusion_score",
    "relevance_score",
    "rule_filter_status",
    "filter_status",
    "filter_reason",
    "emerging_candidate",
    "manual_override",
    "filter_rule_version",
    "first_seen_at",
    "last_seen_at",
]

REVIEW_QUEUE_COLUMNS = [*NEWS_COLUMNS]
REVIEW_COLUMNS = REVIEW_QUEUE_COLUMNS

REJECT_COLUMNS = [
    "news_id",
    "source_post_id",
    "published_at_gmt",
    "title",
    "url",
    "content_class",
    "source_quality",
    "relevance_score",
    "filter_reason",
    "matched_keywords",
    "matched_exclusions",
    "first_seen_at",
    "last_seen_at",
    "filter_rule_version",
]

MANUAL_DECISION_COLUMNS = [
    "news_id",
    "manual_decision",
    "manual_notes",
    "reviewed_at",
    "applied_at",
]

ALIAS_COLUMNS = [
    "ticker",
    "alias",
    "alias_type",
    "priority",
    "allow_standalone",
    "enabled",
    "required_context",
    "excluded_context",
]

ALIAS_SCORE = {
    "high": {"title": 8, "excerpt": 6},
    "medium": {"title": 6, "excerpt": 4},
    "low": {"title": 3, "excerpt": 2},
}

SAFE_TICKER_SCORE = {"title": 6, "excerpt": 3}
PRIORITY_RANK = {"low": 1, "medium": 2, "high": 3}
EMERGING_GENERIC_TERMS = {"ai accelerator"}


@dataclass(frozen=True)
class FilterConfig:
    """
    Loaded V1 filter configuration.
    """

    companies: pd.DataFrame
    aliases: pd.DataFrame
    keywords: pd.DataFrame
    exclusions: pd.DataFrame
    manual_decisions: pd.DataFrame


def _priority_rank(value: object) -> int:
    return PRIORITY_RANK.get(str(value).strip().lower(), 0)


def _enabled(df: pd.DataFrame) -> pd.Series:
    if "enabled" not in df.columns:
        return pd.Series([True] * len(df), index=df.index)

    return df["enabled"].map(csv_bool)


def _normalize_aliases(
    companies: pd.DataFrame,
    aliases: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate, normalize, and de-duplicate alias configuration.
    """

    valid_tickers = set(companies["ticker"].dropna().astype(str).str.upper())
    work = aliases.copy()
    work["ticker"] = work["ticker"].fillna("").astype(str).str.strip().str.upper()
    work["alias"] = work["alias"].fillna("").astype(str).str.strip()
    work["priority"] = work["priority"].fillna("low").astype(str).str.strip().str.lower()
    work["alias_type"] = (
        work["alias_type"].fillna("").astype(str).str.strip().str.lower()
    )
    work["normalized_alias"] = work["alias"].map(normalize_match_text)
    work["priority_rank"] = work["priority"].map(_priority_rank)
    work = work[
        (work["ticker"].isin(valid_tickers))
        & (work["alias"] != "")
        & (work["normalized_alias"] != "")
        & work["enabled"].map(csv_bool)
    ].copy()

    work = work.sort_values(
        by=["ticker", "normalized_alias", "priority_rank"],
        ascending=[True, True, False],
    )

    duplicate_count = work.duplicated(
        subset=["ticker", "normalized_alias"],
        keep="first",
    ).sum()

    if duplicate_count:
        print(f"Warning: removed {duplicate_count} duplicate company alias row(s).")

    work = work.drop_duplicates(
        subset=["ticker", "normalized_alias"],
        keep="first",
    )

    missing_tickers = valid_tickers - set(work["ticker"])

    if missing_tickers:
        print(
            "Warning: company_master ticker(s) missing enabled news aliases: "
            + ", ".join(sorted(missing_tickers))
        )

    return work.reindex(columns=[*ALIAS_COLUMNS, "normalized_alias", "priority_rank"])


def _normalize_keywords(keywords: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize keyword rows and drop duplicate normalized keys.
    """

    work = keywords[keywords["enabled"].map(csv_bool)].copy()
    work["normalized_keyword"] = work["keyword"].fillna("").map(normalize_match_text)
    work = work[work["normalized_keyword"] != ""].copy()
    duplicate_columns = [
        "theme",
        "subtheme",
        "normalized_keyword",
        "keyword_type",
    ]
    duplicate_count = work.duplicated(subset=duplicate_columns, keep="first").sum()

    if duplicate_count:
        print(f"Warning: removed {duplicate_count} duplicate news keyword row(s).")

    return work.drop_duplicates(subset=duplicate_columns, keep="first")


def _active_manual_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    work = decisions.reindex(columns=MANUAL_DECISION_COLUMNS).copy()

    if work.empty:
        return pd.DataFrame(columns=[*MANUAL_DECISION_COLUMNS, "_row_order"])

    work["_row_order"] = range(len(work))
    work["news_id"] = work["news_id"].fillna("").astype(str).str.strip()
    work["manual_decision"] = (
        work["manual_decision"].fillna("").astype(str).str.strip().str.lower()
    )
    active = work[work["manual_decision"] != ""].copy()

    if active.empty:
        return active

    missing_news_id = active["news_id"] == ""

    if missing_news_id.any():
        raise ValueError("Manual decision row(s) with non-empty decision need news_id.")

    invalid = sorted(set(active["manual_decision"]) - {"keep", "reject"})

    if invalid:
        raise ValueError(f"Invalid manual decision value(s): {invalid}")

    active["_reviewed_at_sort"] = pd.to_datetime(
        active["reviewed_at"],
        errors="coerce",
        utc=True,
    )
    active["_has_reviewed_at"] = active["_reviewed_at_sort"].notna().astype(int)
    active = active.sort_values(
        by=["news_id", "_has_reviewed_at", "_reviewed_at_sort", "_row_order"],
        na_position="first",
    )

    return active.drop_duplicates(subset=["news_id"], keep="last")


def load_manual_decisions(path: Path = MANUAL_DECISIONS_FILE) -> pd.DataFrame:
    """
    Load active manual keep/reject decisions with stable schema and validation.
    """

    decisions = read_csv_safe(path, MANUAL_DECISION_COLUMNS)
    active = _active_manual_decisions(decisions)

    if active.empty:
        return pd.DataFrame(columns=MANUAL_DECISION_COLUMNS)

    return active.reindex(columns=MANUAL_DECISION_COLUMNS)


def load_filter_config(
    base_dir: Path = BASE_DIR,
    manual_decisions_file: Path | None = None,
) -> FilterConfig:
    """
    Load company, alias, keyword, exclusion, and manual decision config files.
    """

    master_dir = base_dir / "data" / "master"
    news_dir = base_dir / "data" / "news"
    companies = read_csv_safe(
        master_dir / "company_master.csv",
        [
            "ticker",
            "company",
            "sector",
            "industry_group",
            "theme",
            "subtheme",
            "supply_chain_layer",
            "business_quality_score",
        ],
    )
    aliases = read_csv_safe(master_dir / "company_aliases.csv", ALIAS_COLUMNS)
    keywords = read_csv_safe(
        master_dir / "news_keywords.csv",
        [
            "theme",
            "subtheme",
            "keyword",
            "keyword_type",
            "weight",
            "enabled",
        ],
    )
    exclusions = read_csv_safe(
        master_dir / "news_exclusions.csv",
        [
            "keyword",
            "reason",
            "weight",
            "enabled",
        ],
    )
    manual_decisions = load_manual_decisions(
        manual_decisions_file or news_dir / "news_manual_decisions.csv"
    )

    return FilterConfig(
        companies=companies,
        aliases=_normalize_aliases(companies, aliases),
        keywords=_normalize_keywords(keywords),
        exclusions=exclusions,
        manual_decisions=manual_decisions,
    )


def _as_text(value: object) -> str:
    return clean_html_text(value)


def _split_context(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []

    return [
        normalize_match_text(part)
        for part in str(value).split("|")
        if normalize_match_text(part)
    ]


def _alias_pattern(alias: str) -> re.Pattern[str]:
    words = normalize_match_text(alias).split()
    separator = r"[\W_]+"
    pattern = separator.join(re.escape(word) for word in words)

    return re.compile(rf"(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])", re.IGNORECASE)


def _token_spans(original_text: str) -> list[tuple[str, int, int]]:
    return [
        (match.group(0), match.start(), match.end())
        for match in re.finditer(r"[A-Za-z0-9]+", original_text)
    ]


def _nearby_token_text(
    original_text: str,
    match_start: int,
    match_end: int,
    window_tokens: int = ALIAS_CONTEXT_WINDOW_TOKENS,
) -> str:
    tokens = _token_spans(original_text)
    overlapping = [
        index
        for index, (_, start, end) in enumerate(tokens)
        if start < match_end and end > match_start
    ]

    if not overlapping:
        return original_text[max(0, match_start - 120) : match_end + 120]

    start_index = max(0, overlapping[0] - window_tokens)
    end_index = min(len(tokens), overlapping[-1] + window_tokens + 1)

    return " ".join(token for token, _, _ in tokens[start_index:end_index])


def is_micron_measurement_context(
    original_text: str,
    match_start: int,
    match_end: int,
) -> bool:
    """
    Identify micron as a length-unit reference near a specific alias hit.
    """

    nearby = original_text[max(0, match_start - 80) : match_end + 80].lower()
    measurement_patterns = [
        r"\b\d+(?:\.\d+)?\s*[- ]?microns?\b",
        r"\bmicrons?[- ]scale\b",
        r"\bmicrons?[- ]level\b",
        r"\bmicrons?\s+l\s*/?\s*s\b",
        r"\bmicrons?\s+(?:line|feature|process|node|geometry|pitch)\b",
    ]

    return any(re.search(pattern, nearby) for pattern in measurement_patterns)


def alias_matches(
    alias_row: pd.Series,
    original_text: str,
) -> bool:
    """
    Return whether one alias row matches normalized text and context rules.
    """

    text = clean_html_text(original_text)
    normalized_text = normalize_match_text(text)
    normalized_alias = str(alias_row.get("normalized_alias", "") or "")
    alias = str(alias_row.get("alias", "") or "")
    ticker = str(alias_row.get("ticker", "") or "").strip().upper()
    matches = list(_alias_pattern(alias).finditer(text))

    for excluded in _split_context(alias_row.get("excluded_context", "")):
        if contains_normalized_phrase(normalized_text, excluded):
            return False

    required_context = _split_context(alias_row.get("required_context", ""))

    for match in matches:
        if (
            ticker == "MU"
            and normalized_alias == "micron"
            and is_micron_measurement_context(text, match.start(), match.end())
        ):
            continue

        if required_context:
            nearby_text = normalize_match_text(
                _nearby_token_text(text, match.start(), match.end())
            )

            if not any(
                contains_normalized_phrase(nearby_text, context)
                for context in required_context
            ):
                continue

        elif not csv_bool(alias_row.get("allow_standalone", True)):
            continue

        return True

    return False


def _field_match_score(
    keyword: str,
    title: str,
    excerpt: str,
    taxonomy: str,
    weight: int,
) -> tuple[int, int, list[str]]:
    """
    Score one ordinary theme keyword across title, excerpt, and taxonomy.
    """

    theme_score = 0
    taxonomy_score = 0
    fields: list[str] = []

    if contains_phrase(title, keyword):
        theme_score += 2 * weight
        fields.append("title")

    if contains_phrase(excerpt, keyword):
        theme_score += weight
        fields.append("excerpt")

    if contains_phrase(taxonomy, keyword):
        taxonomy_score += min(weight, 2)
        fields.append("taxonomy")

    return theme_score, taxonomy_score, fields


def _match_aliases(
    title: str,
    excerpt: str,
    config: FilterConfig,
) -> tuple[int, set[str], set[str]]:
    """
    Match manually approved aliases, taking one max score per ticker per field.
    """

    title_scores: dict[str, int] = {}
    excerpt_scores: dict[str, int] = {}
    title_aliases: dict[str, str] = {}
    excerpt_aliases: dict[str, str] = {}

    for _, row in config.aliases.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        alias = str(row.get("alias", "")).strip()
        alias_type = str(row.get("alias_type", "")).strip().lower()
        priority = str(row.get("priority", "")).strip().lower()

        if not ticker or not alias:
            continue

        scoring = SAFE_TICKER_SCORE if alias_type == "safe_ticker" else ALIAS_SCORE.get(
            priority,
            ALIAS_SCORE["low"],
        )

        if alias_matches(row, title):
            score = scoring["title"]

            if score > title_scores.get(ticker, 0):
                title_scores[ticker] = score
                title_aliases[ticker] = alias

        if alias_matches(row, excerpt):
            score = scoring["excerpt"]

            if score > excerpt_scores.get(ticker, 0):
                excerpt_scores[ticker] = score
                excerpt_aliases[ticker] = alias

    matched_tickers = set(title_scores) | set(excerpt_scores)
    matched_aliases = {
        f"{ticker}:{alias}"
        for ticker, alias in {**title_aliases, **excerpt_aliases}.items()
    }
    score = sum(title_scores.values()) + sum(excerpt_scores.values())

    return score, matched_tickers, matched_aliases


def _match_keywords(
    title: str,
    excerpt: str,
    taxonomy: str,
    config: FilterConfig,
) -> tuple[int, int, set[str], set[str], set[str], bool, set[str]]:
    """
    Match theme keywords and emerging-candidate anchors.
    """

    theme_score = 0
    taxonomy_score = 0
    matched_themes: set[str] = set()
    matched_subthemes: set[str] = set()
    matched_keywords: set[str] = set()
    contributing_keywords: set[str] = set()
    has_scope_anchor = False
    has_bottleneck = False
    all_text = " ".join([title, excerpt, taxonomy])

    for _, row in config.keywords.iterrows():
        theme = str(row.get("theme", "")).strip()
        subtheme = str(row.get("subtheme", "")).strip()
        keyword = str(row.get("keyword", "")).strip()
        keyword_type = str(row.get("keyword_type", "")).strip()
        weight = safe_int(row.get("weight"), 0)

        if not keyword:
            continue

        if keyword_type == "scope_anchor":
            if contains_phrase(all_text, keyword):
                has_scope_anchor = True
                matched_themes.add(theme)
                matched_subthemes.add(subtheme)
                matched_keywords.add(keyword)
            continue

        if keyword_type == "bottleneck":
            if contains_phrase(all_text, keyword):
                has_bottleneck = True
                matched_themes.add(theme)
                matched_subthemes.add(subtheme)
                matched_keywords.add(keyword)
            continue

        keyword_theme_score, keyword_taxonomy_score, fields = _field_match_score(
            keyword=keyword,
            title=title,
            excerpt=excerpt,
            taxonomy=taxonomy,
            weight=weight,
        )

        if fields:
            matched_themes.add(theme)
            matched_subthemes.add(subtheme)
            matched_keywords.add(keyword)
            contributing_keywords.add(keyword)
            theme_score += keyword_theme_score
            taxonomy_score += keyword_taxonomy_score

    return (
        theme_score,
        taxonomy_score,
        matched_themes,
        matched_subthemes,
        matched_keywords,
        has_scope_anchor and has_bottleneck,
        contributing_keywords,
    )


def _match_exclusions(
    title: str,
    excerpt: str,
    config: FilterConfig,
) -> tuple[int, set[str]]:
    """
    Match negative scoring terms.
    """

    score = 0
    matched: set[str] = set()
    text = " ".join([title, excerpt])
    enabled_exclusions = config.exclusions[_enabled(config.exclusions)]

    for _, row in enabled_exclusions.iterrows():
        keyword = str(row.get("keyword", "")).strip()
        weight = safe_int(row.get("weight"), 0)

        if keyword and contains_phrase(text, keyword):
            score += weight
            matched.add(keyword)

    return score, matched


def _classify_status(
    company_score: int,
    theme_score: int,
    relevance_score: int,
    emerging_candidate: bool,
    contributing_keywords: set[str],
) -> tuple[str, str, bool]:
    """
    Convert scores into keep/review/reject status.
    """

    generic_only = (
        emerging_candidate
        and company_score < 6
        and contributing_keywords
        and {
            normalize_match_text(keyword)
            for keyword in contributing_keywords
        }.issubset(EMERGING_GENERIC_TERMS)
    )

    if company_score >= 6:
        return "keep", "direct_company_match", emerging_candidate

    if theme_score >= 8 and not generic_only:
        return "keep", "strong_theme_match", emerging_candidate

    if relevance_score >= 10 and not generic_only:
        return "keep", "high_combined_relevance", emerging_candidate

    if emerging_candidate:
        return "review", "emerging_candidate", True

    if 4 <= relevance_score < 10:
        return "review", "moderate_relevance", False

    return "reject", "low_relevance", False


def classify_content(
    title: str,
    category_names: list[str],
    tag_names: list[str],
) -> tuple[str, str]:
    """
    Classify source content type and evidence quality.
    """

    title_text = normalize_match_text(title)
    taxonomy_text = normalize_match_text(" ".join(category_names + tag_names))
    roundup_title_patterns = [
        "week in review",
        "blog review",
        "technical paper roundup",
        "paper roundup",
        "research roundup",
        "research bits",
        "industry roundup",
        "news roundup",
    ]

    if any(
        contains_normalized_phrase(title_text, pattern)
        for pattern in roundup_title_patterns
    ):
        return "roundup", "B"

    if contains_normalized_phrase(taxonomy_text, "technical papers"):
        return "technical_paper", "B"

    if (
        contains_normalized_phrase(taxonomy_text, "white papers")
        or contains_normalized_phrase(taxonomy_text, "whitepapers")
        or contains_normalized_phrase(taxonomy_text, "whitepaper")
    ):
        return "whitepaper", "C"

    if contains_normalized_phrase(taxonomy_text, "blogs"):
        return "vendor_blog", "C"

    if contains_normalized_phrase(taxonomy_text, "opinion"):
        return "opinion", "B"

    if contains_normalized_phrase(taxonomy_text, "videos"):
        return "video", "C"

    if contains_normalized_phrase(taxonomy_text, "round tables"):
        return "editorial", "B"

    if (
        contains_normalized_phrase(taxonomy_text, "top stories")
        or contains_normalized_phrase(taxonomy_text, "news")
    ):
        return "editorial", "A"

    return "unknown", "C"


def filter_article(
    article: dict[str, Any],
    config: FilterConfig | None = None,
) -> dict[str, Any]:
    """
    Score and classify one normalized article row.
    """

    if config is None:
        config = load_filter_config()

    title = _as_text(article.get("title", ""))
    excerpt = _as_text(article.get("excerpt", ""))
    taxonomy = " ".join(
        [
            _as_text(article.get("category_names", "")),
            _as_text(article.get("tag_names", "")),
        ]
    )

    alias_score, alias_tickers, alias_aliases = _match_aliases(
        title=title,
        excerpt=excerpt,
        config=config,
    )
    (
        theme_score,
        taxonomy_score,
        matched_themes,
        matched_subthemes,
        matched_keywords,
        emerging_candidate,
        contributing_keywords,
    ) = _match_keywords(
        title=title,
        excerpt=excerpt,
        taxonomy=taxonomy,
        config=config,
    )
    exclusion_score, matched_exclusions = _match_exclusions(
        title=title,
        excerpt=excerpt,
        config=config,
    )

    company_score = alias_score
    relevance_score = company_score + theme_score + taxonomy_score + exclusion_score
    filter_status, filter_reason, emerging_candidate = _classify_status(
        company_score=company_score,
        theme_score=theme_score,
        relevance_score=relevance_score,
        emerging_candidate=emerging_candidate,
        contributing_keywords=contributing_keywords,
    )

    return {
        "matched_tickers": join_values(alias_tickers),
        "matched_aliases": join_values(alias_aliases),
        "matched_themes": join_values(matched_themes),
        "matched_subthemes": join_values(matched_subthemes),
        "matched_keywords": join_values(matched_keywords),
        "matched_exclusions": join_values(matched_exclusions),
        "company_score": company_score,
        "theme_score": theme_score,
        "taxonomy_score": taxonomy_score,
        "exclusion_score": exclusion_score,
        "relevance_score": relevance_score,
        "rule_filter_status": filter_status,
        "filter_status": filter_status,
        "filter_reason": filter_reason,
        "emerging_candidate": str(emerging_candidate),
        "manual_override": "False",
        "filter_rule_version": FILTER_RULE_VERSION,
    }


def article_from_post(
    post: dict[str, Any],
    category_map: dict[int, str],
    tag_map: dict[int, str],
    seen_at: str,
) -> dict[str, Any]:
    """
    Convert a WordPress post object into the stable news CSV schema.
    """

    source_post_id = str(post.get("id", "")).strip()
    category_ids = [
        int(value)
        for value in post.get("categories", [])
        if str(value).strip().lstrip("-").isdigit()
    ]
    tag_ids = [
        int(value)
        for value in post.get("tags", [])
        if str(value).strip().lstrip("-").isdigit()
    ]
    category_names = [
        category_map[category_id]
        for category_id in category_ids
        if category_id in category_map
    ]
    tag_names = [tag_map[tag_id] for tag_id in tag_ids if tag_id in tag_map]
    content_class, source_quality = classify_content(
        clean_html_text(post.get("title", "")),
        category_names,
        tag_names,
    )

    return {
        "news_id": f"semiengineering_{source_post_id}",
        "source_id": "semiengineering",
        "source_post_id": source_post_id,
        "published_at_local": str(post.get("date", "") or ""),
        "published_at_gmt": normalize_wp_gmt_datetime(post.get("date_gmt", "")),
        "modified_at_gmt": normalize_wp_gmt_datetime(post.get("modified_gmt", "")),
        "title": clean_html_text(post.get("title", "")),
        "excerpt": clean_html_text(post.get("excerpt", "")),
        "url": normalize_url(str(post.get("link", "") or "")),
        "author_id": str(post.get("author", "") or ""),
        "category_ids": join_values(set(category_ids)),
        "category_names": join_values(set(category_names)),
        "tag_ids": join_values(set(tag_ids)),
        "tag_names": join_values(set(tag_names)),
        "content_class": content_class,
        "source_quality": source_quality,
        "first_seen_at": seen_at,
        "last_seen_at": seen_at,
    }


def apply_manual_overrides(
    rows: pd.DataFrame,
    manual_decisions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply persisted manual keep/reject decisions to classified rows.
    """

    if rows.empty or manual_decisions.empty:
        return rows

    valid_decisions = manual_decisions[
        manual_decisions["manual_decision"].isin(["keep", "reject"])
    ]

    if valid_decisions.empty:
        return rows

    decision_by_id = (
        valid_decisions.drop_duplicates(subset=["news_id"], keep="last")
        .set_index("news_id")["manual_decision"]
        .to_dict()
    )
    updated = rows.copy()

    for index, row in updated.iterrows():
        decision = decision_by_id.get(str(row.get("news_id", "")))

        if decision is None:
            continue

        updated.at[index, "filter_status"] = decision
        updated.at[index, "filter_reason"] = f"manual_{decision}"
        updated.at[index, "manual_override"] = "True"

    return updated


def rows_from_posts(
    posts: list[dict[str, Any]],
    category_map: dict[int, str],
    tag_map: dict[int, str],
    seen_at: str,
    config: FilterConfig | None = None,
) -> pd.DataFrame:
    """
    Convert and classify API posts.
    """

    if config is None:
        config = load_filter_config()

    rows: list[dict[str, Any]] = []

    for post in posts:
        article = article_from_post(
            post=post,
            category_map=category_map,
            tag_map=tag_map,
            seen_at=seen_at,
        )
        article.update(filter_article(article, config=config))
        rows.append(article)

    df = pd.DataFrame(rows, columns=NEWS_COLUMNS)
    return apply_manual_overrides(df, config.manual_decisions)


def split_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split full rows by final status.
    """

    keep = rows[rows["filter_status"].isin(["keep", "manual_keep"])].copy()
    review = rows[rows["filter_status"] == "review"].copy()
    reject = rows[rows["filter_status"].isin(["reject", "manual_reject"])].copy()

    return keep, review, reject


def _first_seen_map(*frames: pd.DataFrame) -> dict[str, str]:
    values: dict[str, list[str]] = {}

    for frame in frames:
        if frame.empty or "first_seen_at" not in frame.columns:
            continue

        for _, row in frame.iterrows():
            news_id = str(row.get("news_id", "")).strip()
            first_seen = str(row.get("first_seen_at", "")).strip()

            if not news_id or not first_seen:
                continue

            values.setdefault(news_id, []).append(first_seen)

    return {news_id: sorted(seen_values)[0] for news_id, seen_values in values.items()}


def assert_no_cross_status_overlap(
    keep_df: pd.DataFrame,
    review_df: pd.DataFrame,
    reject_df: pd.DataFrame,
) -> None:
    """
    Ensure no news_id exists in more than one status file.
    """

    ids = {
        "keep": set(keep_df.get("news_id", pd.Series(dtype=str)).dropna().astype(str)),
        "review": set(
            review_df.get("news_id", pd.Series(dtype=str)).dropna().astype(str)
        ),
        "reject": set(
            reject_df.get("news_id", pd.Series(dtype=str)).dropna().astype(str)
        ),
    }
    overlaps = {
        "keep_review": ids["keep"] & ids["review"],
        "keep_reject": ids["keep"] & ids["reject"],
        "review_reject": ids["review"] & ids["reject"],
    }
    actual = {key: value for key, value in overlaps.items() if value}

    if actual:
        raise RuntimeError(f"Cross-status news_id overlap detected: {actual}")


def reconcile_news_statuses(
    rows: pd.DataFrame,
    history_path: Path,
    review_path: Path,
    reject_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reconcile processed rows into mutually exclusive status files.
    """

    existing_keep = read_csv_safe(history_path, NEWS_COLUMNS)
    existing_review = read_csv_safe(review_path, REVIEW_COLUMNS)
    existing_reject = read_csv_safe(reject_path, REJECT_COLUMNS)
    processed_ids = set(rows["news_id"].dropna().astype(str))
    first_seen = _first_seen_map(existing_keep, existing_review, existing_reject, rows)

    remaining_keep = existing_keep[~existing_keep["news_id"].isin(processed_ids)]
    remaining_review = existing_review[~existing_review["news_id"].isin(processed_ids)]
    remaining_reject = existing_reject[~existing_reject["news_id"].isin(processed_ids)]

    new_keep, new_review, new_reject = split_rows(rows)

    for frame in (new_keep, new_review, new_reject):
        if frame.empty:
            continue
        frame["first_seen_at"] = frame["news_id"].map(first_seen).fillna(
            frame["first_seen_at"]
        )

    keep_df = pd.concat([remaining_keep, new_keep], ignore_index=True)
    review_df = pd.concat([remaining_review, new_review], ignore_index=True)
    reject_df = pd.concat(
        [remaining_reject, new_reject.reindex(columns=REJECT_COLUMNS)],
        ignore_index=True,
    )

    if not keep_df.empty:
        keep_df = keep_df.drop_duplicates(subset=["news_id"], keep="last")
        keep_df = keep_df.sort_values(
            by=["published_at_gmt", "news_id"],
            ascending=[False, True],
            na_position="last",
        )

    if not review_df.empty:
        review_df = review_df.drop_duplicates(subset=["news_id"], keep="last")
        review_df = review_df.sort_values(
            by=["published_at_gmt", "news_id"],
            ascending=[False, True],
            na_position="last",
        )

    if not reject_df.empty:
        reject_df = reject_df.drop_duplicates(subset=["news_id"], keep="last")
        reject_df = reject_df.sort_values(
            by=["published_at_gmt", "news_id"],
            ascending=[False, True],
            na_position="last",
        )

    keep_df = keep_df.reindex(columns=NEWS_COLUMNS)
    review_df = review_df.reindex(columns=REVIEW_COLUMNS)
    reject_df = reject_df.reindex(columns=REJECT_COLUMNS)

    for frame in (keep_df, review_df, reject_df):
        if not frame.empty and "filter_rule_version" in frame.columns:
            frame["filter_rule_version"] = FILTER_RULE_VERSION

    assert_no_cross_status_overlap(keep_df, review_df, reject_df)
    atomic_write_csv(keep_df, history_path, NEWS_COLUMNS)
    atomic_write_csv(review_df, review_path, REVIEW_COLUMNS)
    atomic_write_csv(reject_df, reject_path, REJECT_COLUMNS)

    return keep_df, review_df, reject_df


def upsert_manual_decisions(
    existing: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Upsert canonical manual review decisions.
    """

    combined = pd.concat([existing, decisions], ignore_index=True)

    if combined.empty:
        return pd.DataFrame(columns=MANUAL_DECISION_COLUMNS)

    active = _active_manual_decisions(combined)

    if active.empty:
        return pd.DataFrame(columns=MANUAL_DECISION_COLUMNS)

    return active.reindex(columns=MANUAL_DECISION_COLUMNS)

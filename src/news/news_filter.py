"""
Rule-based News Discovery V1 filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from news.news_utils import (
    atomic_write_csv,
    clean_html_text,
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

COMPANY_MASTER_FILE = MASTER_DIR / "company_master.csv"
COMPANY_ALIASES_FILE = MASTER_DIR / "company_aliases.csv"
NEWS_KEYWORDS_FILE = MASTER_DIR / "news_keywords.csv"
NEWS_EXCLUSIONS_FILE = MASTER_DIR / "news_exclusions.csv"

FILTER_RULE_VERSION = "news_filter_v1"

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
    "filter_status",
    "filter_reason",
    "emerging_candidate",
    "filter_rule_version",
    "first_seen_at",
    "last_seen_at",
]

REVIEW_COLUMNS = [
    *NEWS_COLUMNS,
    "manual_decision",
    "manual_notes",
    "reviewed_at",
]

REJECT_COLUMNS = [
    "news_id",
    "source_post_id",
    "published_at_gmt",
    "title",
    "url",
    "relevance_score",
    "filter_reason",
    "matched_keywords",
    "matched_exclusions",
    "first_seen_at",
    "last_seen_at",
    "filter_rule_version",
]

ALIAS_SCORE = {
    "high": {"title": 8, "excerpt": 5},
    "medium": {"title": 6, "excerpt": 4},
    "low": {"title": 3, "excerpt": 2},
}

SAFE_TICKER_SCORE = {"title": 6, "excerpt": 3}
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


def load_filter_config(base_dir: Path = BASE_DIR) -> FilterConfig:
    """
    Load company, alias, keyword, and exclusion config files.
    """

    master_dir = base_dir / "data" / "master"
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
    aliases = read_csv_safe(
        master_dir / "company_aliases.csv",
        [
            "ticker",
            "alias",
            "alias_type",
            "priority",
            "allow_standalone",
            "enabled",
        ],
    )
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

    return FilterConfig(
        companies=companies,
        aliases=aliases,
        keywords=keywords,
        exclusions=exclusions,
    )


def _as_text(value: object) -> str:
    return clean_html_text(value)


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


def _match_company_names(
    title: str,
    excerpt: str,
    config: FilterConfig,
) -> tuple[int, set[str], set[str]]:
    """
    Match official company names from company_master.csv.
    """

    score = 0
    matched_tickers: set[str] = set()
    matched_aliases: set[str] = set()

    for _, row in config.companies.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        company = str(row.get("company", "")).strip()

        if not ticker or not company:
            continue

        matched = False

        if contains_phrase(title, company):
            score += 10
            matched = True

        if contains_phrase(excerpt, company):
            score += 6
            matched = True

        if matched:
            matched_tickers.add(ticker)
            matched_aliases.add(f"{ticker}:{company}")

    return score, matched_tickers, matched_aliases


def _match_aliases(
    title: str,
    excerpt: str,
    config: FilterConfig,
) -> tuple[int, set[str], set[str]]:
    """
    Match manually approved aliases.
    """

    score = 0
    matched_tickers: set[str] = set()
    matched_aliases: set[str] = set()

    enabled_aliases = config.aliases[
        config.aliases["enabled"].map(csv_bool)
        & config.aliases["allow_standalone"].map(csv_bool)
    ]

    for _, row in enabled_aliases.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        alias = str(row.get("alias", "")).strip()
        alias_type = str(row.get("alias_type", "")).strip().lower()
        priority = str(row.get("priority", "")).strip().lower()

        if not ticker or not alias:
            continue

        if alias_type == "safe_ticker":
            scoring = SAFE_TICKER_SCORE
        else:
            scoring = ALIAS_SCORE.get(priority, ALIAS_SCORE["low"])

        matched = False

        if contains_phrase(title, alias):
            score += scoring["title"]
            matched = True

        if contains_phrase(excerpt, alias):
            score += scoring["excerpt"]
            matched = True

        if matched:
            matched_tickers.add(ticker)
            matched_aliases.add(f"{ticker}:{alias}")

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

    enabled_keywords = config.keywords[config.keywords["enabled"].map(csv_bool)]

    for _, row in enabled_keywords.iterrows():
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
    enabled_exclusions = config.exclusions[config.exclusions["enabled"].map(csv_bool)]

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

    official_score, official_tickers, official_aliases = _match_company_names(
        title=title,
        excerpt=excerpt,
        config=config,
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

    company_score = official_score + alias_score
    relevance_score = company_score + theme_score + taxonomy_score + exclusion_score
    filter_status, filter_reason, emerging_candidate = _classify_status(
        company_score=company_score,
        theme_score=theme_score,
        relevance_score=relevance_score,
        emerging_candidate=emerging_candidate,
        contributing_keywords=contributing_keywords,
    )

    matched_tickers = official_tickers | alias_tickers
    matched_aliases = official_aliases | alias_aliases

    return {
        "matched_tickers": join_values(matched_tickers),
        "matched_aliases": join_values(matched_aliases),
        "matched_themes": join_values(matched_themes),
        "matched_subthemes": join_values(matched_subthemes),
        "matched_keywords": join_values(matched_keywords),
        "matched_exclusions": join_values(matched_exclusions),
        "company_score": company_score,
        "theme_score": theme_score,
        "taxonomy_score": taxonomy_score,
        "exclusion_score": exclusion_score,
        "relevance_score": relevance_score,
        "filter_status": filter_status,
        "filter_reason": filter_reason,
        "emerging_candidate": str(emerging_candidate),
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
    category_ids = [int(value) for value in post.get("categories", [])]
    tag_ids = [int(value) for value in post.get("tags", [])]
    category_names = [
        category_map[category_id]
        for category_id in category_ids
        if category_id in category_map
    ]
    tag_names = [tag_map[tag_id] for tag_id in tag_ids if tag_id in tag_map]

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
        "first_seen_at": seen_at,
        "last_seen_at": seen_at,
    }


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

    return pd.DataFrame(rows, columns=NEWS_COLUMNS)


def upsert_keep_history(new_keep: pd.DataFrame, history_file: Path) -> pd.DataFrame:
    """
    Upsert kept rows into permanent history by news_id.
    """

    existing = read_csv_safe(history_file, NEWS_COLUMNS)
    combined = pd.concat([existing, new_keep], ignore_index=True)

    if combined.empty:
        result = pd.DataFrame(columns=NEWS_COLUMNS)
    else:
        first_seen = (
            combined.groupby("news_id", dropna=False)["first_seen_at"]
            .first()
            .to_dict()
        )
        combined = combined.drop_duplicates(subset=["news_id"], keep="last")
        combined["first_seen_at"] = combined["news_id"].map(first_seen)
        result = combined.reindex(columns=NEWS_COLUMNS)
        result = result.sort_values(
            by=["published_at_gmt", "news_id"],
            ascending=[False, True],
            na_position="last",
        )

    atomic_write_csv(result, history_file, NEWS_COLUMNS)
    return result


def upsert_review_queue(new_review: pd.DataFrame, review_file: Path) -> pd.DataFrame:
    """
    Upsert review rows while preserving manual review fields.
    """

    existing = read_csv_safe(review_file, REVIEW_COLUMNS)
    manual_by_id = {}

    if not existing.empty:
        for _, row in existing.iterrows():
            manual_by_id[str(row.get("news_id", ""))] = {
                "manual_decision": row.get("manual_decision", ""),
                "manual_notes": row.get("manual_notes", ""),
                "reviewed_at": row.get("reviewed_at", ""),
                "first_seen_at": row.get("first_seen_at", ""),
            }

    review = new_review.copy()

    for column in REVIEW_COLUMNS:
        if column not in review.columns:
            review[column] = ""

    for index, row in review.iterrows():
        news_id = str(row.get("news_id", ""))
        manual = manual_by_id.get(news_id)

        if manual is None:
            continue

        for column in ("manual_decision", "manual_notes", "reviewed_at"):
            review.at[index, column] = manual[column]

        if manual.get("first_seen_at"):
            review.at[index, "first_seen_at"] = manual["first_seen_at"]

    combined = pd.concat([existing, review], ignore_index=True)

    if combined.empty:
        result = pd.DataFrame(columns=REVIEW_COLUMNS)
    else:
        combined = combined.drop_duplicates(subset=["news_id"], keep="last")
        result = combined.reindex(columns=REVIEW_COLUMNS)
        result = result.sort_values(
            by=["published_at_gmt", "news_id"],
            ascending=[False, True],
            na_position="last",
        )

    atomic_write_csv(result, review_file, REVIEW_COLUMNS)
    return result


def upsert_rejected_log(new_reject: pd.DataFrame, reject_file: Path) -> pd.DataFrame:
    """
    Upsert rejected rows into the lightweight rejected log.
    """

    existing = read_csv_safe(reject_file, REJECT_COLUMNS)
    reject = new_reject.reindex(columns=REJECT_COLUMNS)
    combined = pd.concat([existing, reject], ignore_index=True)

    if combined.empty:
        result = pd.DataFrame(columns=REJECT_COLUMNS)
    else:
        first_seen = (
            combined.groupby("news_id", dropna=False)["first_seen_at"]
            .first()
            .to_dict()
        )
        combined = combined.drop_duplicates(subset=["news_id"], keep="last")
        combined["first_seen_at"] = combined["news_id"].map(first_seen)
        result = combined.reindex(columns=REJECT_COLUMNS)
        result = result.sort_values(
            by=["published_at_gmt", "news_id"],
            ascending=[False, True],
            na_position="last",
        )

    atomic_write_csv(result, reject_file, REJECT_COLUMNS)
    return result

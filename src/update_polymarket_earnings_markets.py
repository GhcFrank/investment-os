"""
update_polymarket_earnings_markets.py

作用：
1. 从 Polymarket Gamma API 抓取 earnings tag 下的 active/open events
2. 过滤 beat consensus / estimates 相关 markets
3. 只保留 company_master.csv 里公司相关的 markets
4. 输出 data/polymarket_earnings_markets.csv

注意：
这个脚本不写预测历史。
它只保存投资池内公司相关的 Polymarket earnings markets。
"""

from pathlib import Path
import re

import pandas as pd

from polymarket_utils import (
    BASE_DIR,
    as_json_string,
    build_polymarket_url,
    fetch_gamma_json,
    today_et,
)


OUTPUT_FILE = BASE_DIR / "data" / "polymarket_earnings_markets.csv"
COMPANY_FILE = BASE_DIR / "data" / "company_master.csv"

PAGE_LIMIT = 100
MAX_EVENTS = 5000
EARNINGS_TAG_ID = 1013

BEAT_CONSENSUS_PATTERNS = [
    re.compile(r"\bbeat quarterly earnings\b", re.IGNORECASE),
    re.compile(r"\bbeat earnings\b", re.IGNORECASE),
    re.compile(r"\bbeat .*consensus\b", re.IGNORECASE),
    re.compile(r"\bbeat .*estimates?\b", re.IGNORECASE),
    re.compile(r"\babove .*consensus\b", re.IGNORECASE),
    re.compile(r"\babove .*estimates?\b", re.IGNORECASE),
]


def normalize_text(value: str) -> str:
    """
    Normalize text for simple matching.
    """

    return re.sub(r"\s+", " ", str(value).strip()).lower()


def ticker_matches_question(
    ticker: str,
    question: str,
) -> bool:
    """
    Match ticker as a standalone token.
    """

    return re.search(
        rf"(?<![A-Za-z0-9]){re.escape(ticker)}(?![A-Za-z0-9])",
        question,
        flags=re.IGNORECASE,
    ) is not None


def company_matches_question(
    company: str,
    question: str,
) -> bool:
    """
    Match company name as a substring.
    """

    company_text = normalize_text(company)
    question_text = normalize_text(question)

    if len(company_text) < 3:
        return False

    return company_text in question_text


def matches_company_master(
    market: dict,
    companies: pd.DataFrame,
) -> bool:
    """
    Return True when a market question matches a company in company_master.csv.
    """

    question = str(market.get("question", ""))

    for _, company_row in companies.iterrows():
        ticker = str(company_row["ticker"]).strip().upper()
        company = str(company_row["company"]).strip()

        if ticker_matches_question(ticker, question):
            return True

        if company_matches_question(company, question):
            return True

    return False


def text_blob(event: dict, market: dict | None = None) -> str:
    """
    Build searchable text from an event and optional market.
    """

    values = [
        event.get("slug"),
        event.get("title"),
        event.get("subtitle"),
    ]

    if market is not None:
        values.extend(
            [
                market.get("slug"),
                market.get("question"),
            ]
        )

    return " ".join(str(value) for value in values if value).lower()


def is_beat_consensus_market(event: dict, market: dict) -> bool:
    """
    Return True for earnings beat consensus / estimates style markets.
    """

    blob = text_blob(event, market)

    return any(pattern.search(blob) for pattern in BEAT_CONSENSUS_PATTERNS)


def fetch_active_events() -> list[dict]:
    """
    Fetch active, open Polymarket events with limit/offset pagination.
    """

    events: list[dict] = []
    offset = 0

    while offset < MAX_EVENTS:
        print(f"Fetching Polymarket events offset={offset}...")

        page = fetch_gamma_json(
            "/events",
            params={
                "active": "true",
                "closed": "false",
                "tag_id": EARNINGS_TAG_ID,
                "limit": PAGE_LIMIT,
                "offset": offset,
            },
        )

        if not isinstance(page, list) or not page:
            break

        events.extend(page)

        if len(page) < PAGE_LIMIT:
            break

        offset += PAGE_LIMIT

    return events


def market_to_row(event: dict, market: dict) -> dict:
    """
    Convert an event + market object into the output CSV schema.
    """

    event_slug = event.get("slug", "")
    market_slug = market.get("slug") or event_slug

    return {
        "date": today_et(),
        "market_id": market.get("id", ""),
        "event_id": event.get("id", ""),
        "slug": market_slug,
        "question": market.get("question", ""),
        "end_date": (
            market.get("endDate")
            or market.get("endDateIso")
            or event.get("endDate")
        ),
        "active": market.get("active", event.get("active", "")),
        "closed": market.get("closed", event.get("closed", "")),
        "outcomes": as_json_string(market.get("outcomes")),
        "outcome_prices": as_json_string(
            market.get("outcomePrices")
            or market.get("outcome_prices")
        ),
        "volume": (
            market.get("volumeNum")
            or market.get("volume")
            or event.get("volume")
        ),
        "liquidity": (
            market.get("liquidityNum")
            or market.get("liquidity")
            or event.get("liquidity")
        ),
        "url": build_polymarket_url(event_slug),
    }


def build_market_rows(events: list[dict]) -> list[dict]:
    """
    Extract earnings-related markets from events.
    """

    if not COMPANY_FILE.exists():
        raise FileNotFoundError(f"Cannot find input file: {COMPANY_FILE}")

    companies = pd.read_csv(COMPANY_FILE)

    for column in ["ticker", "company"]:
        if column not in companies.columns:
            raise ValueError(f"company_master.csv must contain column: {column}")

    rows = []
    seen_market_ids = set()

    for event in events:
        markets = event.get("markets") or []

        for market in markets:
            market_id = str(market.get("id", ""))

            if market_id in seen_market_ids:
                continue

            if not is_beat_consensus_market(event, market):
                continue

            if not matches_company_master(market, companies):
                continue

            seen_market_ids.add(market_id)
            rows.append(market_to_row(event, market))

    return rows


def main() -> None:
    """
    主程序入口。
    """

    events = fetch_active_events()
    rows = build_market_rows(events)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df = pd.DataFrame(
        rows,
        columns=[
            "date",
            "market_id",
            "event_id",
            "slug",
            "question",
            "end_date",
            "active",
            "closed",
            "outcomes",
            "outcome_prices",
            "volume",
            "liquidity",
            "url",
        ],
    )

    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"\nSaved {len(output_df)} Polymarket earnings market(s) to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

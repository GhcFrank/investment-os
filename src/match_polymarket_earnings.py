"""
match_polymarket_earnings.py

作用：
1. 读取 company_master.csv
2. 读取 polymarket_earnings_markets.csv
3. 用简单规则匹配公司和 Polymarket earnings markets
4. 输出 polymarket_earnings_watchlist.csv

第一版规则：
- ticker 出现在 question 中
- company name 出现在 question 中

默认 enabled=False，需要人工确认后才进入每日预测跟踪。
"""

import re

import pandas as pd

from polymarket_utils import BASE_DIR


COMPANY_FILE = BASE_DIR / "data" / "company_master.csv"
MARKETS_FILE = BASE_DIR / "data" / "polymarket_earnings_markets.csv"
OUTPUT_FILE = BASE_DIR / "data" / "polymarket_earnings_watchlist.csv"


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


def match_company_to_market(
    ticker: str,
    company: str,
    question: str,
) -> tuple[str, int] | None:
    """
    Return matched_by and match_score when a company matches a market.
    """

    ticker = str(ticker).strip().upper()
    company = str(company).strip()

    if ticker_matches_question(ticker, question):
        return "ticker", 100

    if company_matches_question(company, question):
        return "company", 90

    return None


def main() -> None:
    """
    主程序入口。
    """

    if not COMPANY_FILE.exists():
        raise FileNotFoundError(f"Cannot find input file: {COMPANY_FILE}")

    if not MARKETS_FILE.exists():
        raise FileNotFoundError(f"Cannot find input file: {MARKETS_FILE}")

    companies = pd.read_csv(COMPANY_FILE)
    markets = pd.read_csv(MARKETS_FILE)

    required_company_columns = ["ticker", "company"]
    required_market_columns = ["market_id", "question"]

    for column in required_company_columns:
        if column not in companies.columns:
            raise ValueError(f"company_master.csv must contain column: {column}")

    for column in required_market_columns:
        if column not in markets.columns:
            raise ValueError(
                f"polymarket_earnings_markets.csv must contain column: {column}"
            )

    rows = []
    enabled_by_key = {}

    if OUTPUT_FILE.exists():
        existing_watchlist = pd.read_csv(OUTPUT_FILE)

        if {"ticker", "market_id", "enabled"}.issubset(existing_watchlist.columns):
            for _, existing_row in existing_watchlist.iterrows():
                key = (
                    str(existing_row["ticker"]).strip().upper(),
                    str(existing_row["market_id"]).strip(),
                )
                enabled_by_key[key] = existing_row["enabled"]

    for _, company_row in companies.iterrows():
        ticker = str(company_row["ticker"]).strip().upper()
        company = str(company_row["company"]).strip()

        for _, market_row in markets.iterrows():
            question = str(market_row.get("question", ""))
            match = match_company_to_market(
                ticker,
                company,
                question,
            )

            if match is None:
                continue

            matched_by, match_score = match
            market_id = str(market_row.get("market_id", "")).strip()
            key = (ticker, market_id)

            rows.append(
                {
                    "ticker": ticker,
                    "company": company,
                    "market_id": market_id,
                    "question": question,
                    "matched_by": matched_by,
                    "match_score": match_score,
                    "enabled": enabled_by_key.get(key, False),
                }
            )

    output_df = pd.DataFrame(
        rows,
        columns=[
            "ticker",
            "company",
            "market_id",
            "question",
            "matched_by",
            "match_score",
            "enabled",
        ],
    )

    output_df = output_df.drop_duplicates(
        subset=["ticker", "market_id"],
        keep="first",
    )

    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"\nSaved {len(output_df)} Polymarket watchlist candidate(s) to: {OUTPUT_FILE}")
    print("Review the file and set enabled=True for markets you want to track.")


if __name__ == "__main__":
    main()

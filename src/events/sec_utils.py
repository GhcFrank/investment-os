"""
SEC EDGAR utility functions for Investment OS filing alerts.

This module centralizes SEC request configuration, CSV loading helpers,
company ticker to CIK mapping, and filing URL construction so the alert
script can stay focused on orchestration.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BASE_DIR / ".env"

DATA_DIR = BASE_DIR / "data"
MASTER_DIR = DATA_DIR / "master"
EVENTS_DIR = DATA_DIR / "events"

COMPANY_MASTER_FILE = MASTER_DIR / "company_master.csv"
SEC_COMPANY_MAP_FILE = EVENTS_DIR / "sec_company_map.csv"

SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"

SEC_COMPANY_MAP_COLUMNS = [
    "ticker",
    "company",
    "cik",
    "sec_title",
    "updated_at",
]


def now_timestamp() -> str:
    """
    Return a readable timestamp for SEC output rows.
    """

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_sec_user_agent() -> str:
    """
    Read and validate the SEC User-Agent value from the environment.

    SEC requires callers to identify themselves. The project supports local
    `.env` files through `python-dotenv`, but the value is still read from the
    process environment after loading that file.

    Raises:
        ValueError: If SEC_USER_AGENT is missing or blank.
    """

    load_dotenv(ENV_FILE)

    user_agent = os.getenv("SEC_USER_AGENT", "").strip()

    if not user_agent:
        raise ValueError(
            "Missing SEC_USER_AGENT environment variable. "
            "Set it to a contactable identifier such as "
            "'InvestmentOS ghcgooder@gmail.com'."
        )

    return user_agent


def get_sec_headers() -> dict[str, str]:
    """
    Build the minimum SEC request headers required by this project.
    """

    return {
        "User-Agent": get_sec_user_agent(),
        "Accept-Encoding": "gzip, deflate",
    }


def sec_get_json(url: str) -> dict[str, Any]:
    """
    Fetch a SEC JSON endpoint with retries, backoff, and status validation.

    Requests use a 30-second timeout and retry up to three times after the
    initial attempt, with exponential backoff delays of 2, 4, and 8 seconds.
    HTTP errors are raised through `raise_for_status()` so callers can decide
    whether to continue or fail the whole script.

    Args:
        url: SEC endpoint URL to fetch.

    Returns:
        Parsed JSON payload.

    Raises:
        requests.RequestException: If all retry attempts fail at the request
            or HTTP status layer.
        ValueError: If the response body cannot be decoded as JSON.
    """

    headers = get_sec_headers()
    last_error: Exception | None = None

    retry_delays = [2, 4, 8]

    for attempt_number in range(len(retry_delays) + 1):
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()

            if not isinstance(payload, dict):
                raise ValueError(f"SEC response was not a JSON object: {url}")

            return payload

        except (requests.RequestException, ValueError) as error:
            last_error = error
            print(f"Warning: SEC request failed for {url}: {error}")

            if attempt_number < len(retry_delays):
                time.sleep(retry_delays[attempt_number])

    if last_error is None:
        raise RuntimeError(f"SEC request failed without an error: {url}")

    raise last_error


def read_csv_safe(path: Path, columns: list[str]) -> pd.DataFrame:
    """
    Read a CSV file and tolerate missing or empty files.

    Args:
        path: CSV file path.
        columns: Columns to use when returning an empty DataFrame.

    Returns:
        Loaded DataFrame, or an empty DataFrame with the requested columns.
    """

    if not path.exists():
        return pd.DataFrame(columns=columns)

    try:
        return pd.read_csv(path, dtype=str)

    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)


def read_company_master() -> pd.DataFrame:
    """
    Load the Investment OS company master file.

    Returns:
        DataFrame containing at least `ticker` and `company`.

    Raises:
        FileNotFoundError: If company_master.csv does not exist.
        ValueError: If required columns are missing.
    """

    if not COMPANY_MASTER_FILE.exists():
        raise FileNotFoundError(f"Cannot find company master file: {COMPANY_MASTER_FILE}")

    required_columns = ["ticker", "company"]

    try:
        companies = pd.read_csv(COMPANY_MASTER_FILE, dtype=str)

    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=required_columns)

    for column in required_columns:
        if column not in companies.columns:
            raise ValueError(f"company_master.csv must contain column: {column}")

    return companies


def normalize_cik(cik: Any) -> str:
    """
    Convert a SEC CIK value to a zero-padded 10-character string.
    """

    return str(int(cik)).zfill(10)


def fetch_sec_company_tickers() -> dict[str, dict[str, Any]]:
    """
    Fetch the official SEC ticker to CIK mapping.

    Returns:
        Mapping keyed by upper-case ticker.
    """

    payload = sec_get_json(SEC_COMPANY_TICKERS_URL)
    mapping: dict[str, dict[str, Any]] = {}

    for record in payload.values():
        if not isinstance(record, dict):
            continue

        ticker = str(record.get("ticker", "")).strip().upper()

        if not ticker:
            continue

        mapping[ticker] = record

    return mapping


def build_sec_company_map(companies: pd.DataFrame) -> pd.DataFrame:
    """
    Match project tickers against the official SEC ticker mapping.

    Tickers are upper-cased and matched exactly. Individual misses are printed
    as warnings and do not stop the run.

    Args:
        companies: Company master rows.

    Returns:
        DataFrame containing matched SEC mapping rows.
    """

    sec_tickers = fetch_sec_company_tickers()
    updated_at = now_timestamp()
    rows: list[dict[str, str]] = []

    for _, company_row in companies.iterrows():
        ticker = str(company_row["ticker"]).strip().upper()
        company = str(company_row.get("company", "")).strip()

        if not ticker:
            continue

        sec_record = sec_tickers.get(ticker)

        if sec_record is None:
            print(f"Warning: SEC ticker mapping not found for {ticker}")
            continue

        rows.append(
            {
                "ticker": ticker,
                "company": company,
                "cik": normalize_cik(sec_record["cik_str"]),
                "sec_title": str(sec_record.get("title", "")).strip(),
                "updated_at": updated_at,
            }
        )

    return pd.DataFrame(rows, columns=SEC_COMPANY_MAP_COLUMNS)


def save_sec_company_map(company_map: pd.DataFrame) -> None:
    """
    Save the SEC company map CSV and print its full path.
    """

    SEC_COMPANY_MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    company_map.to_csv(SEC_COMPANY_MAP_FILE, index=False)
    print(f"Saved SEC company map: {SEC_COMPANY_MAP_FILE}")


def build_filing_urls(cik: str, accession_number: str, primary_document: str) -> tuple[str, str]:
    """
    Build SEC Archives document and filing index URLs.

    Args:
        cik: Zero-padded 10-character CIK.
        accession_number: SEC accession number, usually hyphenated.
        primary_document: Primary filing document filename.

    Returns:
        Tuple of `(filing_url, filing_index_url)`.
    """

    cik_directory = str(int(cik))
    accession_directory = accession_number.replace("-", "")
    filing_index_url = (
        f"{SEC_ARCHIVES_BASE_URL}/{cik_directory}/"
        f"{accession_directory}/{accession_number}-index.html"
    )

    if primary_document:
        filing_url = (
            f"{SEC_ARCHIVES_BASE_URL}/{cik_directory}/"
            f"{accession_directory}/{primary_document}"
        )
    else:
        filing_url = filing_index_url

    return filing_url, filing_index_url

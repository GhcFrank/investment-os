"""
Check SEC EDGAR filings and send Investment OS alerts for new important forms.

The first run initializes a baseline from current SEC filings without sending
email. Later runs compare accession numbers and alert only on filings that have
not been seen before.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from events.sec_utils import (  # noqa: E402
    SEC_SUBMISSIONS_URL_TEMPLATE,
    build_filing_urls,
    build_sec_company_map,
    now_timestamp,
    read_company_master,
    read_csv_safe,
    save_sec_company_map,
    sec_get_json,
)
from utils.send_email import send_email  # noqa: E402


SEC_FILINGS_FILE = BASE_DIR / "data" / "events" / "sec_filings.csv"
SEC_ALERT_HISTORY_FILE = BASE_DIR / "data" / "events" / "sec_alert_history.csv"
SEC_INITIALIZED_FLAG_FILE = BASE_DIR / "data" / "events" / "sec_initialized.flag"

IMPORTANT_FORMS = {
    "8-K",
    "8-K/A",
    "10-Q",
    "10-Q/A",
    "10-K",
    "10-K/A",
    "6-K",
    "20-F",
    "20-F/A",
}

SEC_FILING_COLUMNS = [
    "accession_number",
    "ticker",
    "company",
    "cik",
    "form",
    "filing_date",
    "report_date",
    "acceptance_datetime",
    "primary_document",
    "primary_doc_description",
    "filing_url",
    "filing_index_url",
    "first_seen_at",
    "alert_sent",
]

SEC_ALERT_HISTORY_COLUMNS = [
    "sent_at",
    "accession_number",
    "ticker",
    "company",
    "form",
    "filing_date",
    "filing_index_url",
]

SEC_FILINGS_RETENTION_DAYS = 30


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the SEC filings checker.
    """

    parser = argparse.ArgumentParser(
        description="Check SEC EDGAR filings and send alerts for new filings."
    )
    parser.add_argument(
        "--test-alert",
        action="store_true",
        help="Send a simulated SEC filing email without calling SEC or writing CSV files.",
    )

    return parser.parse_args()


def load_existing_filings() -> pd.DataFrame:
    """
    Load the SEC filings CSV, handling missing, empty, and incomplete files.

    Missing schema columns are added as blank values so a partially damaged CSV
    does not crash the alert run. Rows without an accession number are dropped
    because accession number is the unique key for deduplication.
    """

    filings = read_csv_safe(SEC_FILINGS_FILE, SEC_FILING_COLUMNS)
    missing_columns = [
        column
        for column in SEC_FILING_COLUMNS
        if column not in filings.columns
    ]

    if missing_columns:
        print(
            "Warning: SEC filings CSV is missing columns "
            f"{missing_columns}. Missing values will be treated as blank."
        )

        for column in missing_columns:
            filings[column] = ""

    filings = filings.reindex(columns=SEC_FILING_COLUMNS)

    if filings.empty:
        return filings

    accession_numbers = filings["accession_number"].fillna("").astype(str).str.strip()
    filings = filings[accession_numbers != ""]

    return filings.reindex(columns=SEC_FILING_COLUMNS)


def load_alert_history() -> pd.DataFrame:
    """
    Load the SEC alert history CSV, handling missing and empty files.
    """

    return read_csv_safe(SEC_ALERT_HISTORY_FILE, SEC_ALERT_HISTORY_COLUMNS)


def is_sec_baseline_initialized() -> bool:
    """
    Return whether the SEC baseline has already been initialized.

    The current filings CSV is intentionally not used for this decision. The
    CSV is retention-limited to recent filings, so it can legitimately become
    empty after filtering. Treating an empty DataFrame as "first run" would
    suppress the next real alert by saving it as a fresh baseline.
    """

    return SEC_INITIALIZED_FLAG_FILE.exists()


def mark_sec_baseline_initialized() -> None:
    """
    Persist the SEC baseline initialization marker and print its full path.
    """

    SEC_INITIALIZED_FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
    SEC_INITIALIZED_FLAG_FILE.write_text(
        f"initialized_at={now_timestamp()}\n",
        encoding="utf-8",
    )
    print(f"Saved SEC baseline flag: {SEC_INITIALIZED_FLAG_FILE}")


def get_recent_field(recent: dict[str, Any], field: str, index: int) -> str:
    """
    Return a normalized string value from `payload["filings"]["recent"]`.
    """

    values = recent.get(field, [])

    if not isinstance(values, list) or index >= len(values):
        return ""

    value = values[index]

    if value is None or pd.isna(value):
        return ""

    return str(value)


def extract_important_filings(
    company_row: pd.Series,
    payload: dict[str, Any],
) -> list[dict[str, str]]:
    """
    Extract important filing rows from a single SEC submissions payload.

    Args:
        company_row: Row from the SEC company map.
        payload: SEC submissions JSON payload.

    Returns:
        Filing rows matching `IMPORTANT_FORMS`.
    """

    recent = payload.get("filings", {}).get("recent", {})

    if not isinstance(recent, dict):
        return []

    forms = recent.get("form", [])

    if not isinstance(forms, list):
        return []

    ticker = str(company_row["ticker"]).strip().upper()
    company = str(company_row["company"]).strip()
    cik = str(company_row["cik"]).strip().zfill(10)
    first_seen_at = now_timestamp()
    rows: list[dict[str, str]] = []

    for index, raw_form in enumerate(forms):
        form = str(raw_form).strip()

        if form not in IMPORTANT_FORMS:
            continue

        accession_number = get_recent_field(recent, "accessionNumber", index)

        if not accession_number:
            print(f"Warning: skipping {ticker} filing without accession number")
            continue

        primary_document = get_recent_field(recent, "primaryDocument", index)
        filing_url, filing_index_url = build_filing_urls(
            cik=cik,
            accession_number=accession_number,
            primary_document=primary_document,
        )

        # SEC recent filing arrays are parallel lists, so each field is read at
        # the same index as the matching form.
        rows.append(
            {
                "accession_number": accession_number,
                "ticker": ticker,
                "company": company,
                "cik": cik,
                "form": form,
                "filing_date": get_recent_field(recent, "filingDate", index),
                "report_date": get_recent_field(recent, "reportDate", index),
                "acceptance_datetime": get_recent_field(
                    recent,
                    "acceptanceDateTime",
                    index,
                ),
                "primary_document": primary_document,
                "primary_doc_description": get_recent_field(
                    recent,
                    "primaryDocDescription",
                    index,
                ),
                "filing_url": filing_url,
                "filing_index_url": filing_index_url,
                "first_seen_at": first_seen_at,
                "alert_sent": "False",
            }
        )

    return rows


def fetch_company_filings(company_map: pd.DataFrame) -> list[dict[str, str]]:
    """
    Fetch recent important SEC filings for all mapped companies.

    A single-company failure prints a warning and continues. If every company
    request fails, the function raises RuntimeError because the run cannot be
    trusted.
    """

    all_rows: list[dict[str, str]] = []
    success_count = 0

    for _, company_row in company_map.iterrows():
        ticker = str(company_row["ticker"]).strip().upper()
        cik = str(company_row["cik"]).strip().zfill(10)
        url = SEC_SUBMISSIONS_URL_TEMPLATE.format(cik=cik)

        print(f"Checking SEC filings for {ticker} ({cik})...")

        try:
            payload = sec_get_json(url)
            success_count += 1
            all_rows.extend(
                extract_important_filings(
                    company_row=company_row,
                    payload=payload,
                )
            )

        except Exception as error:
            print(f"Warning: failed to fetch SEC filings for {ticker}: {error}")

        time.sleep(0.15)

    if len(company_map) > 0 and success_count == 0:
        raise RuntimeError("All SEC company filing requests failed.")

    return all_rows


def filter_recent_filing_rows(
    filings: list[dict[str, str]],
    retention_days: int = SEC_FILINGS_RETENTION_DAYS,
) -> list[dict[str, str]]:
    """
    Keep only filing rows whose filing date is within the retention window.

    Args:
        filings: Filing dictionaries extracted from SEC payloads.
        retention_days: Number of recent days to retain.

    Returns:
        Filing rows with `filing_date` on or after the cutoff date.
    """

    if not filings:
        return []

    cutoff_date = (datetime.now().date() - timedelta(days=retention_days)).isoformat()

    return [
        filing
        for filing in filings
        if str(filing.get("filing_date", "")) >= cutoff_date
    ]


def filter_recent_filings(
    filings: pd.DataFrame,
    retention_days: int = SEC_FILINGS_RETENTION_DAYS,
) -> pd.DataFrame:
    """
    Keep only saved SEC filings from the most recent retention window.

    Invalid or missing `filing_date` values are dropped because the retention
    boundary cannot be applied safely to them.
    """

    if filings.empty or "filing_date" not in filings.columns:
        return pd.DataFrame(columns=SEC_FILING_COLUMNS)

    cutoff_date = datetime.now().date() - timedelta(days=retention_days)
    updated = filings.copy()
    filing_dates = pd.to_datetime(
        updated["filing_date"],
        errors="coerce",
    ).dt.date

    kept = updated[filing_dates >= cutoff_date]

    return kept.reindex(columns=SEC_FILING_COLUMNS)


def merge_filings(
    existing_filings: pd.DataFrame,
    incoming_filings: list[dict[str, str]],
) -> pd.DataFrame:
    """
    Merge existing and incoming filings using accession number as the unique key.
    """

    incoming_df = pd.DataFrame(incoming_filings, columns=SEC_FILING_COLUMNS)

    if existing_filings.empty:
        combined = incoming_df
    else:
        combined = pd.concat(
            [existing_filings, incoming_df],
            ignore_index=True,
        )

    if combined.empty:
        return pd.DataFrame(columns=SEC_FILING_COLUMNS)

    combined = combined.drop_duplicates(
        subset=["accession_number"],
        keep="first",
    )

    return combined.reindex(columns=SEC_FILING_COLUMNS)


def save_filings(filings: pd.DataFrame) -> None:
    """
    Save SEC filings CSV and print the full output path.
    """

    SEC_FILINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    filings.to_csv(SEC_FILINGS_FILE, index=False)
    print(f"Saved SEC filings: {SEC_FILINGS_FILE}")


def mark_alerts_sent(
    filings: pd.DataFrame,
    new_filings: list[dict[str, str]],
) -> pd.DataFrame:
    """
    Mark newly alerted accession numbers as sent.
    """

    accession_numbers = {
        filing["accession_number"]
        for filing in new_filings
    }

    if filings.empty or not accession_numbers:
        return filings

    updated = filings.copy()
    matched = updated["accession_number"].astype(str).isin(accession_numbers)
    updated.loc[matched, "alert_sent"] = "True"

    return updated


def append_alert_history(new_filings: list[dict[str, str]]) -> None:
    """
    Append successful SEC filing email alerts to alert history.
    """

    if not new_filings:
        return

    sent_at = now_timestamp()
    existing_history = load_alert_history()
    new_history = pd.DataFrame(
        [
            {
                "sent_at": sent_at,
                "accession_number": filing["accession_number"],
                "ticker": filing["ticker"],
                "company": filing["company"],
                "form": filing["form"],
                "filing_date": filing["filing_date"],
                "filing_index_url": filing["filing_index_url"],
            }
            for filing in new_filings
        ],
        columns=SEC_ALERT_HISTORY_COLUMNS,
    )

    combined = pd.concat(
        [existing_history, new_history],
        ignore_index=True,
    )
    combined = combined.drop_duplicates(
        subset=["accession_number"],
        keep="last",
    )

    SEC_ALERT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(SEC_ALERT_HISTORY_FILE, index=False)
    print(f"Updated SEC alert history: {SEC_ALERT_HISTORY_FILE}")


def find_new_filings(
    existing_filings: pd.DataFrame,
    incoming_filings: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    Identify incoming filings whose accession numbers have not been seen before.
    """

    if existing_filings.empty or "accession_number" not in existing_filings.columns:
        existing_accessions: set[str] = set()
    else:
        existing_accessions = set(
            existing_filings["accession_number"].dropna().astype(str)
        )

    return [
        filing
        for filing in incoming_filings
        if filing["accession_number"] not in existing_accessions
    ]


def build_email_body(new_filings: list[dict[str, str]]) -> str:
    """
    Build a plain-text SEC filing alert email grouped by company.
    """

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)

    for filing in new_filings:
        grouped[(filing["ticker"], filing["company"])].append(filing)

    lines = [
        "Investment OS detected new SEC filings:",
        "",
    ]

    for (ticker, company), filings in sorted(grouped.items()):
        lines.append(f"{ticker}")
        lines.append(f"{company}")

        for filing in filings:
            lines.append(f"form: {filing['form']}")
            lines.append(f"filing_date: {filing['filing_date']}")
            lines.append(f"report_date: {filing['report_date']}")
            lines.append(
                f"primary_doc_description: {filing['primary_doc_description']}"
            )
            lines.append(f"filing_index_url: {filing['filing_index_url']}")
            lines.append("")

    lines.append("This alert was generated by Investment OS.")

    return "\n".join(lines)


def send_test_alert() -> None:
    """
    Send a simulated SEC filing alert without SEC API calls or CSV writes.
    """

    sample_filings = [
        {
            "accession_number": "0000000000-26-000001",
            "ticker": "TEST",
            "company": "Test Company",
            "cik": "0000000000",
            "form": "8-K",
            "filing_date": "2026-06-08",
            "report_date": "2026-06-08",
            "acceptance_datetime": "2026-06-08T16:30:00.000Z",
            "primary_document": "test-8k.htm",
            "primary_doc_description": "Simulated current report",
            "filing_url": "https://www.sec.gov/Archives/edgar/data/0/000000000026000001/test-8k.htm",
            "filing_index_url": "https://www.sec.gov/Archives/edgar/data/0/000000000026000001/0000000000-26-000001-index.html",
            "first_seen_at": now_timestamp(),
            "alert_sent": "False",
        }
    ]

    subject = "Investment OS SEC Filing Alert: 1 new filing(s)"
    body = build_email_body(sample_filings)

    send_email(subject=subject, body=body)
    print("Sent simulated SEC filing test alert.")


def run_sec_filing_check() -> None:
    """
    Run the SEC filing check, persist snapshots, and send alerts when needed.
    """

    companies = read_company_master()
    company_map = build_sec_company_map(companies)
    save_sec_company_map(company_map)

    if company_map.empty:
        raise RuntimeError("No companies could be matched to SEC CIK values.")

    existing_filings = filter_recent_filings(load_existing_filings())

    # Do not use existing_filings.empty here. sec_filings.csv is intentionally
    # retention-limited to recent rows, so it may be empty even after the system
    # has already been initialized. Only the flag represents first-run state.
    baseline_mode = not is_sec_baseline_initialized()

    incoming_filings = filter_recent_filing_rows(fetch_company_filings(company_map))
    combined_filings = merge_filings(
        existing_filings=existing_filings,
        incoming_filings=incoming_filings,
    )

    if baseline_mode:
        save_filings(combined_filings)
        mark_sec_baseline_initialized()
        print("Initialized SEC filing baseline. No alerts sent.")
        return

    new_filings = find_new_filings(
        existing_filings=existing_filings,
        incoming_filings=incoming_filings,
    )

    if not new_filings:
        save_filings(combined_filings)
        print("No new SEC filings detected.")
        return

    subject = f"Investment OS SEC Filing Alert: {len(new_filings)} new filing(s)"
    body = build_email_body(new_filings)

    # If sending fails, let the exception propagate. The script exits non-zero
    # and alert_sent stays False for the new accessions.
    send_email(subject=subject, body=body)

    updated_filings = mark_alerts_sent(
        filings=combined_filings,
        new_filings=new_filings,
    )
    save_filings(updated_filings)
    append_alert_history(new_filings)

    print(f"Sent SEC filing email with {len(new_filings)} new filing(s).")


def main() -> None:
    """
    Main CLI entry point for SEC filing checks.
    """

    args = parse_args()

    if args.test_alert:
        send_test_alert()
        return

    run_sec_filing_check()


if __name__ == "__main__":
    main()

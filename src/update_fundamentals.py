from pathlib import Path
from datetime import datetime

import pandas as pd
import yfinance as yf


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "company_master.csv"
OUTPUT_FILE = BASE_DIR / "data" / "fundamentals.csv"


def get_company_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    return {
        "ticker": ticker,
        "market_cap": info.get("marketCap"),
        "revenue": info.get("totalRevenue"),
        "revenue_growth": info.get("revenueGrowth"),
        "gross_margin": info.get("grossMargins"),
        "operating_margin": info.get("operatingMargins"),
        "forward_pe": info.get("forwardPE"),
        "trailing_pe": info.get("trailingPE"),
        "ev_to_revenue": info.get("enterpriseToRevenue"),
        "updated_at": datetime.now().strftime("%Y-%m-%d")
    }


def main():
    companies = pd.read_csv(INPUT_FILE)

    results = []

    for ticker in companies["ticker"]:
        try:
            print(f"Fetching {ticker}")
            row = get_company_data(ticker)
            results.append(row)

        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    df = pd.DataFrame(results)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
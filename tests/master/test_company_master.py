from pathlib import Path
import unittest

import pandas as pd


class TestCompanyMaster(unittest.TestCase):
    def test_company_master_schema_and_uniqueness(self):
        path = Path("data/master/company_master.csv")
        self.assertTrue(path.exists())

        df = pd.read_csv(path)

        expected_cols = [
            "ticker",
            "company",
            "sector",
            "industry_group",
            "theme",
            "subtheme",
            "supply_chain_layer",
            "business_quality_score",
        ]

        self.assertEqual(list(df.columns), expected_cols)
        self.assertFalse(df["ticker"].duplicated().any())

        for column in expected_cols:
            self.assertFalse(df[column].isna().any(), f"{column} has missing values")

        scores = pd.to_numeric(df["business_quality_score"], errors="raise")
        self.assertTrue(scores.between(1, 10).all())

    def test_requested_tickers_are_present(self):
        path = Path("data/master/company_master.csv")
        df = pd.read_csv(path)

        requested = {
            "DELL",
            "CLS",
            "CRWV",
            "IREN",
            "ANET",
            "APH",
            "GLW",
            "CIEN",
            "NOK",
            "STX",
            "STRL",
            "EQIX",
            "NET",
            "NOW",
            "SNOW",
            "APP",
            "CRWD",
            "FLNC",
            "GEV",
            "BE",
            "ETN",
            "VRT",
            "KLAC",
            "RMBS",
            "COHR",
            "LITE",
            "MRVL",
            "KEYS",
            "AEHR",
        }

        existing = set(df["ticker"].astype(str).str.upper().str.strip())
        self.assertTrue(requested.issubset(existing))


if __name__ == "__main__":
    unittest.main()

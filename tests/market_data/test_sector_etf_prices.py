import copy
import tempfile
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from market_data import update_sector_etf_prices as sector_prices


FETCHED_AT = "2026-07-28T20:00:00Z"


class SectorETFConfigTests(unittest.TestCase):
    EXPECTED_FUND_HISTORY_FILENAMES = {
        "XLC": "xlc_communication_services.csv",
        "XLY": "xly_consumer_discretionary.csv",
        "XLP": "xlp_consumer_staples.csv",
        "XLE": "xle_energy.csv",
        "XLF": "xlf_finance.csv",
        "XLV": "xlv_health_care.csv",
        "XLI": "xli_industrials.csv",
        "XLB": "xlb_materials.csv",
        "XLRE": "xlre_real_estate.csv",
        "XLK": "xlk_information_technology.csv",
        "XLU": "xlu_utilities.csv",
        "SOXX": "soxx_semiconductors.csv",
        "IGV": "igv_software.csv",
    }

    def setUp(self):
        self.payload = yaml.safe_load(
            sector_prices.DEFAULT_CONFIG_FILE.read_text(encoding="utf-8")
        )

    def write_config(self, directory: str, payload: dict) -> Path:
        path = Path(directory) / "sector_etfs.yaml"
        path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return path

    def test_loads_explicit_11_primary_and_13_leadership_universes(self):
        self.payload["etfs"][0]["ticker"] = " xlc "
        with tempfile.TemporaryDirectory() as tmp:
            config = sector_prices.load_sector_etf_config(
                self.write_config(tmp, self.payload)
            )
        self.assertEqual(len(config.etfs), 13)
        self.assertEqual(config.etfs[0].ticker, "XLC")
        self.assertEqual(len({etf.ticker for etf in config.etfs}), 13)
        self.assertEqual(len({etf.sector_id for etf in config.etfs}), 13)
        self.assertEqual(len(config.primary_sector_etfs), 11)
        self.assertEqual(len(config.leadership_etfs), 13)
        self.assertEqual(len(config.state_street_etfs), 11)
        self.assertEqual(len(config.ishares_etfs), 2)
        primary_tickers = {etf.ticker for etf in config.primary_sector_etfs}
        leadership_tickers = {etf.ticker for etf in config.leadership_etfs}
        self.assertTrue({"SOXX", "IGV"}.issubset(leadership_tickers))
        self.assertTrue({"SOXX", "IGV"}.isdisjoint(primary_tickers))
        self.assertEqual(
            {
                etf.ticker: etf.fund_history_filename
                for etf in config.etfs
            },
            self.EXPECTED_FUND_HISTORY_FILENAMES,
        )
        self.assertEqual(
            len({etf.fund_history_filename for etf in config.etfs}),
            13,
        )
        xlf = next(etf for etf in config.etfs if etf.ticker == "XLF")
        self.assertEqual(xlf.sector_id, "financials")
        self.assertEqual(xlf.fund_history_filename, "xlf_finance.csv")
        soxx = next(etf for etf in config.etfs if etf.ticker == "SOXX")
        igv = next(etf for etf in config.etfs if etf.ticker == "IGV")
        self.assertEqual(soxx.ishares_product_id, 239705)
        self.assertEqual(igv.ishares_product_id, 239771)
        self.assertEqual(soxx.fund_history_filename, "soxx_semiconductors.csv")
        self.assertEqual(igv.fund_history_filename, "igv_software.csv")
        self.assertEqual(soxx.metrics_filename, "soxx_semiconductors.csv")
        self.assertEqual(igv.metrics_filename, "igv_software.csv")
        self.assertEqual(soxx.classification_level, "industry")
        self.assertEqual(igv.classification_level, "industry")
        self.assertIn(
            "{ticker_lower}",
            config.state_street_nav_history_url_template,
        )

    def test_missing_required_field_raises(self):
        del self.payload["etfs"][0]["sector_name_cn"]
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_config(tmp, self.payload)
            with self.assertRaisesRegex(ValueError, "sector_name_cn"):
                sector_prices.load_sector_etf_config(path)

    def test_industry_overlays_cannot_change_primary_sector_input(self):
        config = sector_prices.load_sector_etf_config()
        original_rows = pd.DataFrame(
            {
                "ticker": config.primary_sector_tickers,
                "value": range(len(config.primary_sector_tickers)),
            }
        )
        with_overlays = pd.concat(
            [
                original_rows,
                pd.DataFrame(
                    {
                        "ticker": ["SOXX", "IGV"],
                        "value": [999, 1000],
                    }
                ),
            ],
            ignore_index=True,
        )
        selected = with_overlays.loc[
            with_overlays["ticker"].isin(config.primary_sector_tickers)
        ].reset_index(drop=True)
        original_concentration = (
            original_rows["value"].pow(2).sum()
        )
        selected_concentration = selected["value"].pow(2).sum()
        pd.testing.assert_frame_equal(selected, original_rows)
        self.assertEqual(selected_concentration, original_concentration)
        self.assertEqual(len(selected), 11)
        self.assertTrue({"SOXX", "IGV"}.isdisjoint(selected["ticker"]))

    def test_duplicate_ticker_or_sector_raises(self):
        self.payload["etfs"][1]["ticker"] = self.payload["etfs"][0]["ticker"]
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_config(tmp, self.payload)
            with self.assertRaisesRegex(ValueError, "duplicate ticker"):
                sector_prices.load_sector_etf_config(path)

        self.setUp()
        self.payload["etfs"][1]["sector_id"] = self.payload["etfs"][0][
            "sector_id"
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_config(tmp, self.payload)
            with self.assertRaisesRegex(ValueError, "duplicate sector_id"):
                sector_prices.load_sector_etf_config(path)

    def test_fund_history_filename_must_be_unique_safe_lowercase_csv(self):
        invalid_names = [
            "../xlf.csv",
            "nested/xlf.csv",
            r"nested\xlf.csv",
            "/tmp/xlf.csv",
            "XLF_finance.csv",
            "xlf-finance.csv",
            "xlf_finance.txt",
        ]
        for invalid_name in invalid_names:
            with self.subTest(filename=invalid_name):
                payload = copy.deepcopy(self.payload)
                payload["etfs"][4]["fund_history_filename"] = invalid_name
                with tempfile.TemporaryDirectory() as tmp:
                    path = self.write_config(tmp, payload)
                    with self.assertRaisesRegex(
                        ValueError,
                        "safe lowercase .csv basename",
                    ):
                        sector_prices.load_sector_etf_config(path)

        payload = copy.deepcopy(self.payload)
        payload["etfs"][1]["fund_history_filename"] = payload["etfs"][0][
            "fund_history_filename"
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_config(tmp, payload)
            with self.assertRaisesRegex(
                ValueError,
                "duplicate fund_history_filename",
            ):
                sector_prices.load_sector_etf_config(path)

    def test_ishares_product_id_must_be_positive(self):
        soxx = next(
            etf
            for etf in self.payload["etfs"]
            if etf["ticker"] == "SOXX"
        )
        soxx["ishares_product_id"] = 0
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_config(tmp, self.payload)
            with self.assertRaisesRegex(ValueError, "positive"):
                sector_prices.load_sector_etf_config(path)


class SectorETFPriceNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.config = sector_prices.load_sector_etf_config()
        self.xlc = self.config.etfs[0]
        self.xly = self.config.etfs[1]

    def test_single_ticker_normalization_cleans_invalid_and_duplicate_rows(self):
        raw = pd.DataFrame(
            {
                "Open": [10, 11, 12, 13],
                "High": [11, 12, 13, float("inf")],
                "Low": [9, 10, 11, 12],
                "Close": [10.5, 11.5, 12.5, None],
                "Adj Close": [10.25, 11.25, 12.25, 13.25],
                "Volume": [100, 200, 300, 400],
            },
            index=pd.to_datetime(
                ["2026-07-27", "2026-07-28", "2026-07-28", "2026-07-29"]
            ),
        )
        normalized = sector_prices.normalize_sector_etf_prices(
            raw,
            self.xlc,
            fetched_at_utc=FETCHED_AT,
        )
        self.assertEqual(list(normalized.columns), sector_prices.PRICE_COLUMNS)
        self.assertEqual(list(normalized["date"]), ["2026-07-27", "2026-07-28"])
        self.assertEqual(list(normalized["close"]), [10.5, 12.5])
        self.assertEqual(list(normalized["adj_close"]), [10.25, 12.25])
        self.assertEqual(set(normalized["ticker"]), {"XLC"})

    def test_both_yfinance_multiindex_layouts_become_long_format(self):
        ticker_first = pd.DataFrame(
            [
                [10, 10.5, 10.25, 100, 20, 20.5, 20.25, 200],
            ],
            index=pd.to_datetime(["2026-07-28"]),
            columns=pd.MultiIndex.from_product(
                [["XLC", "XLY"], ["Open", "Close", "Adj Close", "Volume"]]
            ),
        )
        normalized = sector_prices.normalize_sector_etf_prices(
            ticker_first,
            [self.xlc, self.xly],
            fetched_at_utc=FETCHED_AT,
        )
        self.assertEqual(list(normalized["ticker"]), ["XLC", "XLY"])
        self.assertEqual(list(normalized["close"]), [10.5, 20.5])
        self.assertEqual(list(normalized["adj_close"]), [10.25, 20.25])

        price_first = pd.DataFrame(
            [[10.5, 20.5, 10.25, 20.25]],
            index=pd.to_datetime(["2026-07-28"]),
            columns=pd.MultiIndex.from_product(
                [["Close", "Adj Close"], ["XLC", "XLY"]]
            ),
        )
        normalized = sector_prices.normalize_sector_etf_prices(
            price_first,
            [self.xlc, self.xly],
            fetched_at_utc=FETCHED_AT,
        )
        self.assertEqual(list(normalized["close"]), [10.5, 20.5])
        self.assertEqual(list(normalized["adj_close"]), [10.25, 20.25])


class SectorETFPriceDownloadTests(unittest.TestCase):
    def test_failure_isolated_and_overlap_and_auto_adjust_are_explicit(self):
        full_config = sector_prices.load_sector_etf_config()
        config = replace(
            full_config,
            etfs=full_config.etfs[:2],
            leadership_tickers=("XLC", "XLY"),
        )

        class FakeYahoo:
            def __init__(self):
                self.calls = []

            def download(self, **kwargs):
                self.calls.append(kwargs)
                if kwargs["tickers"] == "XLY":
                    raise RuntimeError("temporary failure")
                return pd.DataFrame(
                    {
                        "Open": [10],
                        "High": [11],
                        "Low": [9],
                        "Close": [10.5],
                        "Adj Close": [10.25],
                        "Volume": [100],
                    },
                    index=pd.to_datetime(["2026-07-28"]),
                )

        yahoo = FakeYahoo()
        result = sector_prices.download_sector_etf_prices(
            config,
            latest_dates={"XLC": date(2026, 7, 28)},
            end_date="2026-07-28",
            yf_module=yahoo,
            max_attempts=1,
            sleep_func=lambda _: None,
            fetched_at_utc=FETCHED_AT,
        )
        self.assertEqual(result.succeeded, ["XLC"])
        self.assertEqual(set(result.errors), {"XLY"})
        xlc_call, xly_call = yahoo.calls
        self.assertEqual(xlc_call["start"], "2026-07-23")
        self.assertEqual(xlc_call["end"], "2026-07-29")
        self.assertFalse(xlc_call["auto_adjust"])
        self.assertEqual(xly_call["period"], "max")


def price_row(date_value: str, ticker: str, close: float, fetched_at: str) -> dict:
    return {
        "date": date_value,
        "ticker": ticker,
        "sector_id": "sector",
        "sector_name": "Sector",
        "sector_name_cn": "板块",
        "open": close - 1,
        "high": close + 1,
        "low": close - 2,
        "close": close,
        "adj_close": close - 0.25,
        "volume": 100,
        "source": "yahoo_finance",
        "fetched_at_utc": fetched_at,
    }


class SectorETFPriceUpsertTests(unittest.TestCase):
    def test_insert_update_and_repeat_are_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "prices.csv"
            first = pd.DataFrame(
                [
                    price_row("2026-07-27", "XLY", 20, "first"),
                    price_row("2026-07-27", "XLC", 10, "first"),
                ]
            )
            first_stats = sector_prices.upsert_sector_etf_prices(first, output)
            incremental = pd.DataFrame(
                [
                    price_row("2026-07-27", "XLC", 11, "second"),
                    price_row("2026-07-28", "XLC", 12, "second"),
                ]
            )
            second_stats = sector_prices.upsert_sector_etf_prices(
                incremental,
                output,
            )
            second_mtime = output.stat().st_mtime_ns
            repeat_stats = sector_prices.upsert_sector_etf_prices(
                incremental,
                output,
            )
            repeat_mtime = output.stat().st_mtime_ns
            saved = pd.read_csv(output)
            header = output.read_text(encoding="utf-8").splitlines()[0]

        self.assertEqual(first_stats.inserted, 2)
        self.assertEqual(second_stats.inserted, 1)
        self.assertEqual(second_stats.updated, 1)
        self.assertEqual(repeat_stats.inserted, 0)
        self.assertEqual(repeat_stats.updated, 0)
        self.assertFalse(repeat_stats.file_written)
        self.assertEqual(second_mtime, repeat_mtime)
        self.assertEqual(len(saved), 3)
        self.assertFalse(saved.duplicated(["date", "ticker"]).any())
        self.assertEqual(
            list(zip(saved["date"], saved["ticker"])),
            [
                ("2026-07-27", "XLC"),
                ("2026-07-27", "XLY"),
                ("2026-07-28", "XLC"),
            ],
        )
        self.assertEqual(header, ",".join(sector_prices.PRICE_COLUMNS))


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from signals import build_sector_etf_metrics as metrics


def one_ticker_prices(rows, ticker="XLF"):
    return pd.DataFrame(
        [
            {
                "date": date_value,
                "ticker": ticker,
                "adj_close": adj_close,
                **extra,
            }
            for date_value, adj_close, extra in rows
        ]
    )


class CalendarDayReturnTests(unittest.TestCase):
    def test_exact_weekend_and_holiday_targets_use_backward_asof(self):
        prices = one_ticker_prices(
            [
                ("2024-01-05", 100.0, {}),
                ("2024-01-08", 101.0, {}),
                ("2024-01-10", 101.5, {}),
                ("2024-01-12", 102.0, {}),
                ("2024-01-16", 103.0, {}),
                ("2024-02-05", 104.0, {}),
                ("2024-02-06", 105.0, {}),
                ("2024-02-09", 106.0, {}),
                ("2024-02-14", 107.0, {}),
                ("2024-04-04", 110.0, {}),
                ("2024-04-05", 120.0, {}),
            ]
        )

        result_90d = metrics.calculate_calendar_day_adj_close_return(
            prices,
            90,
        )
        rows_90d = pd.concat(
            [prices[["date"]].reset_index(drop=True), result_90d],
            axis=1,
        ).set_index("date")
        self.assertEqual(
            rows_90d.at[
                "2024-04-04",
                "reference_date_90d",
            ].strftime("%Y-%m-%d"),
            "2024-01-05",
        )
        # 2024-04-05 - 90 days is Saturday 2024-01-06. The following
        # Monday must not be selected.
        self.assertEqual(
            rows_90d.at[
                "2024-04-05",
                "reference_date_90d",
            ].strftime("%Y-%m-%d"),
            "2024-01-05",
        )

        result_30d = metrics.calculate_calendar_day_adj_close_return(
            prices,
            30,
        )
        rows_30d = pd.concat(
            [prices[["date"]].reset_index(drop=True), result_30d],
            axis=1,
        ).set_index("date")
        # 2024-02-09 - 30 days is the exact trading date 2024-01-10.
        self.assertEqual(
            rows_30d.at[
                "2024-02-09",
                "reference_date_30d",
            ].strftime("%Y-%m-%d"),
            "2024-01-10",
        )
        # Saturday and Sunday targets both use the prior Friday.
        self.assertEqual(
            rows_30d.at[
                "2024-02-05",
                "reference_date_30d",
            ].strftime("%Y-%m-%d"),
            "2024-01-05",
        )
        self.assertEqual(
            rows_30d.at[
                "2024-02-06",
                "reference_date_30d",
            ].strftime("%Y-%m-%d"),
            "2024-01-05",
        )
        # 2024-02-14 - 30 days is the 2024-01-15 market holiday.
        self.assertEqual(
            rows_30d.at[
                "2024-02-14",
                "reference_date_30d",
            ].strftime("%Y-%m-%d"),
            "2024-01-12",
        )

    def test_return_formula_saves_gain_and_loss_not_price_ratio(self):
        prices = one_ticker_prices(
            [
                ("2024-01-01", 100.0, {}),
                ("2024-01-02", 100.0, {}),
                ("2024-03-31", 125.0, {}),
                ("2024-04-01", 80.0, {}),
            ]
        )
        result = metrics.calculate_calendar_day_adj_close_return(prices, 90)
        rows = pd.concat(
            [prices[["date"]].reset_index(drop=True), result],
            axis=1,
        ).set_index("date")
        self.assertAlmostEqual(
            rows.at["2024-03-31", "adj_close_return_90d"],
            0.25,
        )
        self.assertNotEqual(
            rows.at["2024-03-31", "adj_close_return_90d"],
            1.25,
        )
        self.assertAlmostEqual(
            rows.at["2024-04-01", "adj_close_return_90d"],
            -0.20,
        )

    def test_builder_uses_adj_close_and_never_raw_close(self):
        prices = one_ticker_prices(
            [
                ("2024-01-01", 80.0, {"close": 100.0}),
                ("2024-03-31", 100.0, {"close": 110.0}),
            ]
        )
        result = metrics.build_one_sector_etf_metrics(prices)
        current = result.loc[result["date"].eq("2024-03-31")].iloc[0]
        self.assertAlmostEqual(current["adj_close_return_90d"], 0.25)
        self.assertNotAlmostEqual(current["adj_close_return_90d"], 0.10)
        self.assertAlmostEqual(current["reference_adj_close_90d"], 80.0)

    def test_all_horizons_are_calendar_days_not_row_shifts(self):
        prices = one_ticker_prices(
            [
                ("2020-01-02", 10.0, {}),
                ("2020-05-01", 20.0, {}),
                ("2020-09-01", 40.0, {}),
                ("2020-12-08", 60.0, {}),
                ("2021-01-08", 80.0, {}),
            ]
        )
        result = metrics.build_one_sector_etf_metrics(prices)
        current = result.loc[result["date"].eq("2021-01-08")].iloc[0]
        self.assertEqual(current["reference_date_250d"], "2020-05-01")
        self.assertEqual(current["reference_date_90d"], "2020-09-01")
        self.assertEqual(current["reference_date_30d"], "2020-12-08")
        self.assertAlmostEqual(current["adj_close_return_250d"], 3.0)
        self.assertAlmostEqual(current["adj_close_return_90d"], 1.0)
        self.assertAlmostEqual(
            current["adj_close_return_30d"],
            80.0 / 60.0 - 1.0,
        )

    def test_insufficient_history_keeps_rows_and_nulls_all_reference_fields(self):
        prices = one_ticker_prices(
            [
                ("2024-01-02", 100.0, {}),
                ("2024-01-20", 105.0, {}),
            ]
        )
        result = metrics.build_one_sector_etf_metrics(prices)
        self.assertEqual(len(result), 2)
        for horizon in metrics.SECTOR_ETF_RETURN_HORIZONS:
            self.assertTrue(result[f"reference_date_{horizon}d"].isna().all())
            self.assertTrue(
                result[f"reference_adj_close_{horizon}d"].isna().all()
            )
            self.assertTrue(
                result[f"adj_close_return_{horizon}d"].isna().all()
            )

    def test_two_etfs_never_share_reference_history(self):
        config = metrics.load_sector_etf_config()
        selected = replace(
            config,
            etfs=tuple(
                etf for etf in config.etfs if etf.ticker in {"XLF", "XLK"}
            ),
        )
        prices = pd.concat(
            [
                one_ticker_prices(
                    [
                        ("2024-01-01", 100.0, {}),
                        ("2024-03-31", 125.0, {}),
                    ],
                    ticker="XLF",
                ),
                one_ticker_prices(
                    [
                        ("2024-01-02", 50.0, {}),
                        ("2024-04-01", 40.0, {}),
                    ],
                    ticker="XLK",
                ),
            ],
            ignore_index=True,
        )
        result = metrics.build_all_sector_etf_metrics(prices, selected)
        xlf = result["XLF"].iloc[-1]
        xlk = result["XLK"].iloc[-1]
        self.assertEqual(xlf["reference_date_90d"], "2024-01-01")
        self.assertEqual(xlf["reference_adj_close_90d"], 100.0)
        self.assertAlmostEqual(xlf["adj_close_return_90d"], 0.25)
        self.assertEqual(xlk["reference_date_90d"], "2024-01-02")
        self.assertEqual(xlk["reference_adj_close_90d"], 50.0)
        self.assertAlmostEqual(xlk["adj_close_return_90d"], -0.20)


class SectorETFPriceValidationTests(unittest.TestCase):
    def setUp(self):
        self.valid = pd.DataFrame(
            {
                "date": ["2025-01-02", "2025-01-03"],
                "ticker": ["XLF", "XLF"],
                "adj_close": [100.0, 101.0],
            }
        )

    def test_each_required_column_is_mandatory_and_close_is_not_fallback(self):
        for column in metrics.REQUIRED_PRICE_COLUMNS:
            with self.subTest(column=column):
                invalid = self.valid.drop(columns=column)
                if column == "adj_close":
                    invalid["close"] = [90.0, 91.0]
                with self.assertRaisesRegex(
                    metrics.SectorETFMetricsValidationError,
                    column,
                ):
                    metrics.validate_sector_etf_prices(invalid)

    def test_duplicate_date_ticker_after_case_normalization_raises(self):
        invalid = self.valid.copy()
        invalid["date"] = ["2025-01-02", "2025-01-02"]
        invalid["ticker"] = ["xlf", " XLF "]
        with self.assertRaisesRegex(
            metrics.SectorETFMetricsValidationError,
            "duplicate date \\+ ticker",
        ):
            metrics.validate_sector_etf_prices(invalid)

    def test_invalid_date_empty_ticker_and_future_date_raise(self):
        invalid_date = self.valid.copy()
        invalid_date.loc[0, "date"] = "not-a-date"
        with self.assertRaisesRegex(
            metrics.SectorETFMetricsValidationError,
            "invalid date",
        ):
            metrics.validate_sector_etf_prices(invalid_date)

        empty_ticker = self.valid.copy()
        empty_ticker.loc[0, "ticker"] = " "
        with self.assertRaisesRegex(
            metrics.SectorETFMetricsValidationError,
            "empty ticker",
        ):
            metrics.validate_sector_etf_prices(empty_ticker)

        future = self.valid.copy()
        future.loc[1, "date"] = "2026-01-03"
        with self.assertRaisesRegex(
            metrics.SectorETFMetricsValidationError,
            "future market date",
        ):
            metrics.validate_sector_etf_prices(
                future,
                today=date(2026, 1, 2),
            )

    def test_invalid_zero_negative_nan_and_infinite_adj_close_raise(self):
        invalid_values = [
            "not-a-number",
            0,
            -1,
            np.nan,
            np.inf,
            -np.inf,
        ]
        for invalid_value in invalid_values:
            with self.subTest(adj_close=invalid_value):
                invalid = self.valid.copy()
                invalid["adj_close"] = invalid["adj_close"].astype(object)
                invalid.loc[0, "adj_close"] = invalid_value
                with self.assertRaises(
                    metrics.SectorETFMetricsValidationError
                ):
                    metrics.validate_sector_etf_prices(invalid)

    def test_numeric_strings_are_converted_and_tickers_are_uppercase(self):
        values = self.valid.copy()
        values["ticker"] = [" xlf ", "xlf"]
        values["adj_close"] = ["100.5", "101.5"]
        normalized = metrics.validate_sector_etf_prices(values)
        self.assertEqual(set(normalized["ticker"]), {"XLF"})
        self.assertEqual(list(normalized["adj_close"]), [100.5, 101.5])

    def test_configured_ticker_set_must_match_input(self):
        with self.assertRaisesRegex(
            metrics.SectorETFMetricsValidationError,
            "missing configured ticker",
        ):
            metrics.validate_sector_etf_prices(
                self.valid,
                expected_tickers={"XLF", "XLK"},
            )
        with self.assertRaisesRegex(
            metrics.SectorETFMetricsValidationError,
            "unconfigured ticker",
        ):
            metrics.validate_sector_etf_prices(
                self.valid,
                expected_tickers={"XLK"},
            )


class SectorETFMetricsOutputTests(unittest.TestCase):
    def setUp(self):
        self.prices = one_ticker_prices(
            [
                ("2024-01-02", 100.0, {}),
                ("2025-01-02", 125.0, {}),
            ]
        )
        self.built = metrics.build_one_sector_etf_metrics(self.prices)

    def test_schema_order_sorting_no_index_and_blank_null_serialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "xlf_finance.csv"
            self.assertTrue(metrics.write_metrics_atomic(self.built, output))
            header = output.read_text(encoding="utf-8").splitlines()[0]
            saved = pd.read_csv(output, keep_default_na=False)

        self.assertEqual(list(saved.columns), metrics.METRICS_COLUMNS)
        self.assertEqual(header, ",".join(metrics.METRICS_COLUMNS))
        self.assertEqual(
            metrics.SECTOR_ETF_RETURN_HORIZONS,
            (250, 90, 30),
        )
        self.assertFalse(
            any("120d" in column for column in metrics.METRICS_COLUMNS)
        )
        self.assertNotIn("Unnamed: 0", saved.columns)
        self.assertEqual(list(saved["date"]), ["2024-01-02", "2025-01-02"])
        self.assertEqual(saved.at[0, "reference_date_90d"], "")
        self.assertEqual(saved.at[0, "reference_adj_close_90d"], "")
        self.assertEqual(saved.at[0, "adj_close_return_90d"], "")

    def test_production_metrics_source_has_no_120_day_semantics(self):
        source = Path(metrics.__file__).read_text(encoding="utf-8")
        for obsolete in (
            "adj_close_return_120d",
            "reference_date_120d",
            "reference_adj_close_120d",
        ):
            self.assertNotIn(obsolete, source)

    def test_identical_content_is_not_rewritten_and_mtime_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "xlf_finance.csv"
            self.assertTrue(metrics.write_metrics_atomic(self.built, output))
            first_content = output.read_bytes()
            first_mtime = output.stat().st_mtime_ns
            self.assertFalse(metrics.write_metrics_atomic(self.built, output))
            second_content = output.read_bytes()
            second_mtime = output.stat().st_mtime_ns

        self.assertEqual(first_content, second_content)
        self.assertEqual(first_mtime, second_mtime)

    def test_failed_atomic_write_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "xlf_finance.csv"
            output.write_text("old-content\n", encoding="utf-8")
            with (
                patch.object(
                    metrics,
                    "atomic_write_csv",
                    side_effect=RuntimeError("write failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "write failed"),
            ):
                metrics.write_metrics_atomic(self.built, output)
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "old-content\n",
            )

    def test_full_orchestration_writes_configured_filenames_and_is_idempotent(
        self,
    ):
        config = metrics.load_sector_etf_config()
        rows = []
        for index, etf in enumerate(config.etfs):
            rows.extend(
                [
                    {
                        "date": "2020-01-02",
                        "ticker": etf.ticker.lower(),
                        "adj_close": 100.0 + index,
                    },
                    {
                        "date": "2021-01-08",
                        "ticker": etf.ticker,
                        "adj_close": 200.0 + index,
                    },
                ]
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            price_file = root / "sector_etf_prices.csv"
            output_dir = root / "metrics"
            pd.DataFrame(rows).to_csv(price_file, index=False)

            first = metrics.run_sector_etf_metrics_update(
                price_file=price_file,
                output_dir=output_dir,
            )
            expected_names = {
                etf.fund_history_filename for etf in config.etfs
            }
            actual_names = {path.name for path in output_dir.glob("*.csv")}
            first_mtimes = {
                path.name: path.stat().st_mtime_ns
                for path in output_dir.glob("*.csv")
            }
            second = metrics.run_sector_etf_metrics_update(
                price_file=price_file,
                output_dir=output_dir,
            )
            second_mtimes = {
                path.name: path.stat().st_mtime_ns
                for path in output_dir.glob("*.csv")
            }
            saved_xlf = pd.read_csv(output_dir / "xlf_finance.csv")

        self.assertEqual(first.succeeded, 11)
        self.assertEqual(first.failed, 0)
        self.assertEqual(first.files_written, 11)
        self.assertEqual(actual_names, expected_names)
        self.assertIn("xlc_communication_services.csv", actual_names)
        self.assertIn("xlf_finance.csv", actual_names)
        self.assertIn("xlk_information_technology.csv", actual_names)
        self.assertIn("xlre_real_estate.csv", actual_names)
        self.assertNotIn("xlf.csv", actual_names)
        self.assertNotIn("xlk.csv", actual_names)
        self.assertEqual(len(saved_xlf), 2)
        self.assertEqual(second.files_written, 0)
        self.assertEqual(second.files_unchanged, 11)
        self.assertEqual(first_mtimes, second_mtimes)

    def test_one_etf_write_failure_is_recorded_while_others_continue(self):
        config = metrics.load_sector_etf_config()
        rows = [
            {
                "date": "2025-01-02",
                "ticker": etf.ticker,
                "adj_close": 100.0 + index,
            }
            for index, etf in enumerate(config.etfs)
        ]
        real_write = metrics.write_metrics_atomic

        def write_except_xlf(frame, path):
            if Path(path).name == "xlf_finance.csv":
                raise OSError("simulated XLF failure")
            return real_write(frame, path)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            price_file = root / "sector_etf_prices.csv"
            output_dir = root / "metrics"
            pd.DataFrame(rows).to_csv(price_file, index=False)
            with patch.object(
                metrics,
                "write_metrics_atomic",
                side_effect=write_except_xlf,
            ):
                summary = metrics.run_sector_etf_metrics_update(
                    price_file=price_file,
                    output_dir=output_dir,
                )
            actual_names = {path.name for path in output_dir.glob("*.csv")}

        self.assertEqual(summary.succeeded, 10)
        self.assertEqual(summary.failed, 1)
        self.assertIn("XLF", summary.errors)
        self.assertNotIn("xlf_finance.csv", actual_names)
        self.assertIn("xlk_information_technology.csv", actual_names)

    def test_structural_input_error_aborts_before_any_output(self):
        config = metrics.load_sector_etf_config()
        invalid_rows = [
            {
                "date": "2025-01-02",
                "ticker": etf.ticker,
                "close": 100.0,
            }
            for etf in config.etfs
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            price_file = root / "sector_etf_prices.csv"
            output_dir = root / "metrics"
            pd.DataFrame(invalid_rows).to_csv(price_file, index=False)
            with self.assertRaisesRegex(
                metrics.SectorETFMetricsValidationError,
                "adj_close",
            ):
                metrics.run_sector_etf_metrics_update(
                    price_file=price_file,
                    output_dir=output_dir,
                )
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()

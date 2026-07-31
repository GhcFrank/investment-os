import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import requests
from openpyxl import Workbook

from market_data import update_sector_etf_fund_history as fund_history
from market_data.sector_etf_config import load_sector_etf_config


def workbook_bytes(
    ticker: str = "XLF",
    *,
    include_total_assets: bool = True,
    include_data: bool = True,
    sheet_name: str = "navhist",
    zero_shares: bool = False,
) -> bytes:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = sheet_name
    sheet.append(["Fund Name:", "State Street Test ETF", None, None, None, None])
    sheet.append(["Ticker Symbol:", ticker, None, None, None, None])
    sheet.append([None, None, None, None, None, None])
    headers = ["Date", "NAV", "Shares Outstanding"]
    if include_total_assets:
        headers.append("Total Net Assets")
    headers.extend(["Unused E", "Unused F"])
    sheet.append(headers)
    if include_data:
        sheet.append(
            [
                "27-Jul-2026",
                56.0,
                0 if zero_shares else 100,
                0 if zero_shares else 5600.0,
                None,
                None,
            ]
        )
        sheet.append(["24-Jul-2026", 55.0, "-", "-", None, None])
        sheet.append(["23-Jul-2026", 54.0, None, None, None, None])
        sheet.append([None, None, None, None, None, None])
        sheet.append(
            [
                "The information contained herein is not an offer.",
                None,
                None,
                None,
                None,
                None,
            ]
        )
    output = BytesIO()
    workbook.save(output)
    return output.getvalue()


def history_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=fund_history.FUND_HISTORY_COLUMNS)


class FakeResponse:
    def __init__(self, content: bytes, content_type: str = ""):
        self.content = content
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self):
        return None


class FakeHTTPClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, url, timeout):
        self.calls.append((url, timeout))
        for ticker, response in self.responses.items():
            if f"-{ticker.lower()}.xlsx" in url:
                if isinstance(response, Exception):
                    raise response
                return response
        raise AssertionError(f"Unexpected URL: {url}")


class StateStreetURLTests(unittest.TestCase):
    def test_urls_come_from_config_template(self):
        config = load_sector_etf_config()
        xlf = fund_history.build_state_street_nav_history_url(config, "XLF")
        xlre = fund_history.build_state_street_nav_history_url(config, "xlre")
        self.assertTrue(xlf.endswith("navhist-us-en-xlf.xlsx"))
        self.assertTrue(xlre.endswith("navhist-us-en-xlre.xlsx"))
        with self.assertRaisesRegex(ValueError, "not configured"):
            fund_history.build_state_street_nav_history_url(config, "SOXX")

    def test_configured_output_paths_use_descriptive_safe_basenames(self):
        config = load_sector_etf_config()
        paths = {
            etf.ticker: fund_history.configured_fund_history_path(
                Path("/tmp/history"),
                etf,
            ).name
            for etf in config.state_street_etfs
        }
        self.assertEqual(paths["XLC"], "xlc_communication_services.csv")
        self.assertEqual(paths["XLF"], "xlf_finance.csv")
        self.assertEqual(paths["XLK"], "xlk_information_technology.csv")
        self.assertEqual(paths["XLRE"], "xlre_real_estate.csv")
        self.assertEqual(len(set(paths.values())), 11)
        self.assertNotIn("SOXX", paths)
        self.assertNotIn("IGV", paths)


class StateStreetParserTests(unittest.TestCase):
    def test_real_layout_parses_missing_values_and_filters_legal_text(self):
        parsed = fund_history.parse_state_street_nav_history(
            workbook_bytes(),
            requested_ticker="XLF",
        )
        self.assertEqual(
            list(parsed.columns),
            fund_history.FUND_HISTORY_COLUMNS,
        )
        self.assertEqual(
            list(parsed["date"]),
            ["2026-07-23", "2026-07-24", "2026-07-27"],
        )
        self.assertEqual(list(parsed["nav"]), [54.0, 55.0, 56.0])
        self.assertTrue(pd.isna(parsed.loc[0, "shares_outstanding"]))
        self.assertTrue(pd.isna(parsed.loc[1, "total_net_assets"]))
        self.assertEqual(parsed.loc[2, "shares_outstanding"], 100)
        self.assertEqual(parsed.loc[2, "total_net_assets"], 5600.0)

    def test_ticker_mismatch_raises(self):
        with self.assertRaises(
            fund_history.StateStreetTickerMismatchError
        ):
            fund_history.parse_state_street_nav_history(
                workbook_bytes("XLC"),
                requested_ticker="XLF",
            )

    def test_zero_shares_are_logged_as_missing_while_nav_is_preserved(self):
        parsed = fund_history.parse_state_street_nav_history(
            workbook_bytes("XLC", zero_shares=True),
            requested_ticker="XLC",
        )
        latest = parsed.iloc[-1]
        self.assertEqual(latest["nav"], 56.0)
        self.assertTrue(pd.isna(latest["shares_outstanding"]))
        self.assertEqual(latest["total_net_assets"], 0.0)
        self.assertEqual(
            parsed.attrs["zero_share_rows_normalized"],
            1,
        )

    def test_missing_required_column_raises(self):
        with self.assertRaisesRegex(
            fund_history.StateStreetFileFormatError,
            "missing required columns",
        ):
            fund_history.parse_state_street_nav_history(
                workbook_bytes(include_total_assets=False),
                requested_ticker="XLF",
            )

    def test_html_or_non_xlsx_payload_raises(self):
        for payload in (b"<html>blocked</html>", b"not a workbook"):
            with self.subTest(payload=payload):
                with self.assertRaises(
                    fund_history.StateStreetFileFormatError
                ):
                    fund_history.parse_state_street_nav_history(
                        payload,
                        requested_ticker="XLF",
                    )

    def test_empty_navhist_or_missing_sheet_raises(self):
        with self.assertRaises(
            fund_history.StateStreetDataValidationError
        ):
            fund_history.parse_state_street_nav_history(
                workbook_bytes(include_data=False),
                requested_ticker="XLF",
            )
        with self.assertRaisesRegex(
            fund_history.StateStreetFileFormatError,
            "navhist sheet",
        ):
            fund_history.parse_state_street_nav_history(
                workbook_bytes(sheet_name="other"),
                requested_ticker="XLF",
            )

    def test_download_rejects_html_and_does_not_write_xlsx(self):
        client = FakeHTTPClient(
            {
                "XLF": FakeResponse(
                    b"<html>blocked</html>",
                    "text/html",
                )
            }
        )
        with self.assertRaises(
            fund_history.StateStreetFileFormatError
        ):
            fund_history.download_state_street_nav_history(
                "https://example.test/navhist-us-en-xlf.xlsx",
                ticker="XLF",
                http_client=client,
                max_attempts=1,
                sleep_func=lambda _: None,
            )


class FundHistoryMergeTests(unittest.TestCase):
    def test_merge_preserves_old_and_local_non_null_values(self):
        local = history_frame(
            [
                {
                    "date": "2026-07-22",
                    "nav": 9.0,
                    "shares_outstanding": 90,
                    "total_net_assets": 810.0,
                },
                {
                    "date": "2026-07-23",
                    "nav": 10.0,
                    "shares_outstanding": 100,
                    "total_net_assets": 1000.0,
                },
                {
                    "date": "2026-07-24",
                    "nav": 11.0,
                    "shares_outstanding": None,
                    "total_net_assets": None,
                },
            ]
        )
        remote = history_frame(
            [
                {
                    "date": "2026-07-23",
                    "nav": 10.5,
                    "shares_outstanding": None,
                    "total_net_assets": 1050.0,
                },
                {
                    "date": "2026-07-24",
                    "nav": 11.0,
                    "shares_outstanding": 110,
                    "total_net_assets": 1210.0,
                },
                {
                    "date": "2026-07-25",
                    "nav": 12.0,
                    "shares_outstanding": 120,
                    "total_net_assets": 1440.0,
                },
            ]
        )

        result = fund_history.merge_fund_history(local, remote)
        merged = result.history.set_index("date")
        self.assertEqual(result.inserted_rows, 1)
        self.assertEqual(result.updated_rows, 2)
        self.assertEqual(list(merged.index), [
            "2026-07-22",
            "2026-07-23",
            "2026-07-24",
            "2026-07-25",
        ])
        self.assertEqual(merged.loc["2026-07-23", "nav"], 10.5)
        self.assertEqual(
            merged.loc["2026-07-23", "shares_outstanding"],
            100,
        )
        self.assertEqual(
            merged.loc["2026-07-24", "total_net_assets"],
            1210.0,
        )

        repeated = fund_history.merge_fund_history(result.history, remote)
        self.assertEqual(repeated.inserted_rows, 0)
        self.assertEqual(repeated.updated_rows, 0)
        self.assertTrue(
            fund_history.fund_histories_equal(
                result.history,
                repeated.history,
            )
        )

    def test_first_write_then_identical_remote_does_not_rewrite(self):
        config = load_sector_etf_config()
        etf = next(etf for etf in config.etfs if etf.ticker == "XLF")
        client = FakeHTTPClient(
            {"XLF": FakeResponse(workbook_bytes(), "application/octet-stream")}
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "history"
            first = fund_history.update_one_sector_etf_fund_history(
                etf,
                config,
                output_dir=output_dir,
                http_client=client,
                max_attempts=1,
                sleep_func=lambda _: None,
            )
            with patch.object(
                fund_history,
                "write_fund_history_atomic",
            ) as writer:
                second = fund_history.update_one_sector_etf_fund_history(
                    etf,
                    config,
                    output_dir=output_dir,
                    http_client=client,
                    max_attempts=1,
                    sleep_func=lambda _: None,
                )
            output = fund_history.configured_fund_history_path(output_dir, etf)
            saved = pd.read_csv(output)
            legacy_exists = fund_history.legacy_fund_history_path(
                output_dir,
                etf,
            ).exists()
            xlsx_files = list(output_dir.rglob("*.xlsx"))

        self.assertTrue(first.file_written)
        self.assertFalse(second.file_written)
        self.assertEqual(second.inserted_rows, 0)
        self.assertEqual(second.updated_rows, 0)
        writer.assert_not_called()
        self.assertFalse(saved["date"].duplicated().any())
        self.assertFalse(legacy_exists)
        self.assertEqual(xlsx_files, [])

    def test_clearly_stale_remote_does_not_overwrite_local_file(self):
        config = load_sector_etf_config()
        etf = next(etf for etf in config.etfs if etf.ticker == "XLF")
        local = history_frame(
            [
                {
                    "date": "2026-07-27",
                    "nav": 56.0,
                    "shares_outstanding": 100,
                    "total_net_assets": 5600.0,
                }
            ]
        )
        stale_remote = history_frame(
            [
                {
                    "date": "2026-07-01",
                    "nav": 55.0,
                    "shares_outstanding": 100,
                    "total_net_assets": 5500.0,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            output = fund_history.configured_fund_history_path(
                output_dir,
                etf,
            )
            fund_history.write_fund_history_atomic(local, output)
            original = output.read_bytes()
            with (
                patch.object(
                    fund_history,
                    "download_state_street_nav_history",
                    return_value=b"PK",
                ),
                patch.object(
                    fund_history,
                    "parse_state_street_nav_history",
                    return_value=stale_remote,
                ),
                self.assertRaisesRegex(
                    fund_history.StateStreetDataValidationError,
                    "older than local",
                ),
            ):
                fund_history.update_one_sector_etf_fund_history(
                    etf,
                    config,
                    output_dir=output_dir,
                )
            after = output.read_bytes()

        self.assertEqual(original, after)


class FundHistoryMigrationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_sector_etf_config()
        self.etf = next(
            etf for etf in self.config.etfs
            if etf.ticker == "XLF"
        )
        self.legacy_rows = history_frame(
            [
                {
                    "date": "2026-07-22",
                    "nav": 9.0,
                    "shares_outstanding": 90,
                    "total_net_assets": 810.0,
                },
                {
                    "date": "2026-07-23",
                    "nav": 10.0,
                    "shares_outstanding": 100,
                    "total_net_assets": 1000.0,
                },
            ]
        )
        self.configured_rows = history_frame(
            [
                {
                    "date": "2026-07-23",
                    "nav": 10.5,
                    "shares_outstanding": None,
                    "total_net_assets": 1050.0,
                },
                {
                    "date": "2026-07-24",
                    "nav": 11.0,
                    "shares_outstanding": 110,
                    "total_net_assets": 1210.0,
                },
            ]
        )

    def paths(self, directory: str) -> tuple[Path, Path]:
        legacy = fund_history.legacy_fund_history_path(directory, self.etf)
        configured = fund_history.configured_fund_history_path(
            directory,
            self.etf,
        )
        return legacy, configured

    def test_only_legacy_file_is_validated_and_renamed_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy, configured = self.paths(tmp)
            fund_history.write_fund_history_atomic(self.legacy_rows, legacy)
            original = legacy.read_bytes()
            result = fund_history.migrate_one_sector_etf_fund_history(
                self.etf,
                output_dir=tmp,
            )

            self.assertEqual(result.action, "renamed")
            self.assertFalse(legacy.exists())
            self.assertEqual(configured.read_bytes(), original)

    def test_only_configured_file_is_validated_without_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy, configured = self.paths(tmp)
            fund_history.write_fund_history_atomic(
                self.configured_rows,
                configured,
            )
            original = configured.read_bytes()
            result = fund_history.migrate_one_sector_etf_fund_history(
                self.etf,
                output_dir=tmp,
            )

            self.assertEqual(result.action, "validated")
            self.assertFalse(legacy.exists())
            self.assertEqual(configured.read_bytes(), original)

    def test_both_files_merge_with_configured_non_null_values_preferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy, configured = self.paths(tmp)
            fund_history.write_fund_history_atomic(self.legacy_rows, legacy)
            fund_history.write_fund_history_atomic(
                self.configured_rows,
                configured,
            )
            result = fund_history.migrate_one_sector_etf_fund_history(
                self.etf,
                output_dir=tmp,
            )
            merged = fund_history.load_local_fund_history(
                configured,
            ).set_index("date")

            self.assertEqual(result.action, "merged")
            self.assertFalse(legacy.exists())
            self.assertEqual(
                list(merged.index),
                ["2026-07-22", "2026-07-23", "2026-07-24"],
            )
            self.assertEqual(merged.loc["2026-07-23", "nav"], 10.5)
            self.assertEqual(
                merged.loc["2026-07-23", "shares_outstanding"],
                100,
            )
            self.assertEqual(
                merged.loc["2026-07-23", "total_net_assets"],
                1050.0,
            )

    def test_merge_write_failure_preserves_both_original_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy, configured = self.paths(tmp)
            fund_history.write_fund_history_atomic(self.legacy_rows, legacy)
            fund_history.write_fund_history_atomic(
                self.configured_rows,
                configured,
            )
            original_legacy = legacy.read_bytes()
            original_configured = configured.read_bytes()
            with (
                patch.object(
                    fund_history,
                    "write_fund_history_atomic",
                    side_effect=OSError("disk failure"),
                ),
                self.assertRaisesRegex(OSError, "disk failure"),
            ):
                fund_history.migrate_one_sector_etf_fund_history(
                    self.etf,
                    output_dir=tmp,
                )

            self.assertEqual(legacy.read_bytes(), original_legacy)
            self.assertEqual(configured.read_bytes(), original_configured)


class FundHistoryAtomicWriteTests(unittest.TestCase):
    def test_write_failure_preserves_original_and_removes_temp(self):
        rows = history_frame(
            [
                {
                    "date": "2026-07-27",
                    "nav": 56.0,
                    "shares_outstanding": 100,
                    "total_net_assets": 5600.0,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "target.csv"
            output.write_text("original\n", encoding="utf-8")
            with (
                patch.object(
                    pd.DataFrame,
                    "to_csv",
                    side_effect=OSError("disk failure"),
                ),
                self.assertRaises(OSError),
            ):
                fund_history.write_fund_history_atomic(rows, output)
            remaining = list(Path(tmp).iterdir())
            content = output.read_text(encoding="utf-8")

        self.assertEqual(content, "original\n")
        self.assertEqual(remaining, [output])


class FundHistoryRunTests(unittest.TestCase):
    def test_one_failure_does_not_block_other_ticker(self):
        client = FakeHTTPClient(
            {
                "XLC": FakeResponse(workbook_bytes("XLC")),
                "XLY": requests.ConnectionError("offline"),
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            summary = fund_history.run_sector_etf_fund_history_update(
                output_dir=Path(tmp) / "history",
                tickers=["xlc", "XLY"],
                http_client=client,
                max_attempts=1,
                sleep_func=lambda _: None,
            )
            config = load_sector_etf_config()
            xlc = next(etf for etf in config.etfs if etf.ticker == "XLC")
            xly = next(etf for etf in config.etfs if etf.ticker == "XLY")
            output_dir = Path(tmp) / "history"
            xlc_exists = fund_history.configured_fund_history_path(
                output_dir,
                xlc,
            ).exists()
            xly_exists = fund_history.configured_fund_history_path(
                output_dir,
                xly,
            ).exists()
            legacy_files = [
                fund_history.legacy_fund_history_path(output_dir, etf)
                for etf in (xlc, xly)
                if fund_history.legacy_fund_history_path(
                    output_dir,
                    etf,
                ).exists()
            ]

        self.assertEqual(summary.succeeded, 1)
        self.assertEqual(summary.failed, 1)
        self.assertTrue(xlc_exists)
        self.assertFalse(xly_exists)
        self.assertEqual(legacy_files, [])

    def test_bootstrap_subset_writes_configured_name_not_legacy_name(self):
        client = FakeHTTPClient(
            {"XLF": FakeResponse(workbook_bytes("XLF"))}
        )
        config = load_sector_etf_config()
        etf = next(etf for etf in config.etfs if etf.ticker == "XLF")
        with tempfile.TemporaryDirectory() as tmp:
            summary = fund_history.run_sector_etf_fund_history_update(
                output_dir=tmp,
                tickers=["xlf"],
                bootstrap=True,
                http_client=client,
                max_attempts=1,
                sleep_func=lambda _: None,
            )
            configured_exists = fund_history.configured_fund_history_path(
                tmp,
                etf,
            ).exists()
            legacy_exists = fund_history.legacy_fund_history_path(
                tmp,
                etf,
            ).exists()

        self.assertEqual(summary.mode, "bootstrap")
        self.assertEqual(summary.succeeded, 1)
        self.assertTrue(configured_exists)
        self.assertFalse(legacy_exists)

    def test_invalid_requested_ticker_is_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "NOTREAL"):
                fund_history.run_sector_etf_fund_history_update(
                    output_dir=tmp,
                    tickers=["notreal"],
                )

    def test_production_source_has_no_deprecated_aum_snapshot_reference(self):
        deprecated_name = "".join(
            ("sector_etf_", "aum_", "history.csv")
        )
        source_root = fund_history.BASE_DIR / "src"
        references = [
            path
            for path in source_root.rglob("*.py")
            if deprecated_name in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(references, [])


if __name__ == "__main__":
    unittest.main()

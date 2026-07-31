import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from openpyxl import Workbook

from market_data import update_ishares_etf_fund_history as ishares
from market_data.sector_etf_config import load_sector_etf_config


def spreadsheetml_bytes(rows=None) -> bytes:
    rows = rows or [
        ("Jan 03, 2025", "10.0", "1000"),
        ("Jan 02, 2025", "9.5", "900"),
    ]
    xml_rows = [
        """
        <ss:Row>
          <ss:Cell><ss:Data ss:Type="String">As Of</ss:Data></ss:Cell>
          <ss:Cell><ss:Data ss:Type="String">NAV per Share</ss:Data></ss:Cell>
          <ss:Cell><ss:Data ss:Type="String">Ex-Dividends</ss:Data></ss:Cell>
          <ss:Cell><ss:Data ss:Type="String">Shares Outstanding</ss:Data></ss:Cell>
        </ss:Row>
        """
    ]
    for row_date, nav, shares in rows:
        xml_rows.append(
            f"""
            <ss:Row>
              <ss:Cell><ss:Data ss:Type="String">{row_date}</ss:Data></ss:Cell>
              <ss:Cell><ss:Data ss:Type="Number">{nav}</ss:Data></ss:Cell>
              <ss:Cell><ss:Data ss:Type="String">--</ss:Data></ss:Cell>
              <ss:Cell><ss:Data ss:Type="Number">{shares}</ss:Data></ss:Cell>
            </ss:Row>
            """
        )
    return (
        '<?xml version="1.0"?>'
        '<ss:Workbook xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">'
        '<ss:Worksheet ss:Name="Disclaimers"><ss:Table>'
        '<ss:Row><ss:Cell ss:HRef="https://example.test/?a=1&b=2">'
        '<ss:Data ss:Type="String">Legal</ss:Data></ss:Cell></ss:Row>'
        "</ss:Table></ss:Worksheet>"
        '<ss:Worksheet ss:Name="Historical"><ss:Table>'
        + "".join(xml_rows)
        + "</ss:Table></ss:Worksheet></ss:Workbook>"
    ).encode()


def xlsx_bytes() -> bytes:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Historical"
    sheet.append(["As Of", "NAV per Share", "Ex-Dividends", "Shares Outstanding"])
    sheet.append(["Jan 03, 2025", 10.0, "--", 1000])
    sheet.append(["Jan 02, 2025", 9.5, "--", 900])
    output = BytesIO()
    workbook.save(output)
    return output.getvalue()


def product_html(
    ticker="SOXX",
    *,
    nav="10.00",
    nav_date="Jan 03, 2025",
    assets="$10,000",
    assets_date="as of Jan 03, 2025",
    shares="1,000",
    shares_date="as of Jan 03, 2025",
) -> bytes:
    payload = {
        "@graph": [
            {
                "@type": "FinancialProduct",
                "identifier": [
                    {
                        "@type": "PropertyValue",
                        "propertyID": "ticker",
                        "value": ticker,
                    }
                ],
            },
            {
                "@type": "WebPageElement",
                "name": "KeyDataPointsV3",
                "additionalProperty": [
                    {
                        "@type": "PropertyValue",
                        "name": "NAV as of",
                        "value": nav,
                        "valueReference": {
                            "name": "As of Dates",
                            "value": nav_date,
                        },
                    }
                ],
            },
        ]
    }
    return f"""<!doctype html><html><body>
      <script type="application/ld+json">{json.dumps(payload)}</script>
      <div data-id="keyFundFacts-totalNetAssetsFundLevel-data">{assets}</div>
      <div data-id="keyFundFacts-totalNetAssetsFundLevel-asOf">{assets_date}</div>
      <div data-id="keyFundFacts-sharesOutstanding-data">{shares}</div>
      <div data-id="keyFundFacts-sharesOutstanding-asOf">{shares_date}</div>
    </body></html>""".encode()


class FakeResponse:
    def __init__(self, content, content_type, status_code=200):
        self.content = content
        self.headers = {"Content-Type": content_type}
        self.status_code = status_code

    def raise_for_status(self):
        return None


class FakeHTTPClient:
    def __init__(self, fund_content, page_content):
        self.fund_content = fund_content
        self.page_content = page_content
        self.calls = []

    def get(self, url, timeout, headers):
        self.calls.append((url, timeout, headers))
        if "component=fundDownload" in url:
            return FakeResponse(
                self.fund_content,
                "application/vnd.ms-excel",
            )
        return FakeResponse(self.page_content, "text/html; charset=UTF-8")


class ISharesConfigAndURLTests(unittest.TestCase):
    def test_product_ids_and_official_urls_do_not_cross(self):
        config = load_sector_etf_config()
        by_ticker = {etf.ticker: etf for etf in config.ishares_etfs}
        soxx_download = ishares.build_ishares_fund_download_url(
            config,
            by_ticker["SOXX"],
        )
        igv_download = ishares.build_ishares_fund_download_url(
            config,
            by_ticker["IGV"],
        )
        soxx_page = ishares.build_ishares_product_page_url(
            config,
            by_ticker["SOXX"],
        )
        igv_page = ishares.build_ishares_product_page_url(
            config,
            by_ticker["IGV"],
        )
        self.assertIn("portfolioId=239705", soxx_download)
        self.assertIn("portfolioId=239771", igv_download)
        self.assertIn("/239705/", soxx_page)
        self.assertIn("/239771/", igv_page)
        self.assertNotIn("239771", soxx_download + soxx_page)
        self.assertNotIn("239705", igv_download + igv_page)


class ISharesFormatAndParserTests(unittest.TestCase):
    def test_content_based_format_detection(self):
        self.assertEqual(
            ishares.detect_ishares_download_format(xlsx_bytes()),
            "xlsx",
        )
        self.assertEqual(
            ishares.detect_ishares_download_format(spreadsheetml_bytes()),
            "spreadsheetml",
        )
        for payload, content_type in (
            (b"<html>Access Denied</html>", "application/vnd.ms-excel"),
            (b"", ""),
            (b"\x00\x01unknown", "application/octet-stream"),
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ishares.ISharesFileFormatError):
                    ishares.detect_ishares_download_format(
                        payload,
                        content_type=content_type,
                    )

    def test_xlsx_and_spreadsheetml_use_same_historical_schema(self):
        for payload in (xlsx_bytes(), spreadsheetml_bytes()):
            with self.subTest(signature=payload[:8]):
                history = ishares.parse_ishares_fund_history(
                    payload,
                    ticker="SOXX",
                )
                self.assertEqual(
                    list(history.columns),
                    ishares.FUND_HISTORY_COLUMNS,
                )
                self.assertEqual(
                    list(history["date"]),
                    ["2025-01-02", "2025-01-03"],
                )
                self.assertEqual(list(history["nav"]), [9.5, 10.0])
                self.assertEqual(
                    list(history["shares_outstanding"]),
                    [900, 1000],
                )
                self.assertTrue(history["total_net_assets"].isna().all())

    def test_snapshot_cleans_fields_and_requires_matching_dates(self):
        snapshot = ishares.parse_ishares_product_snapshot(
            product_html(),
            requested_ticker="SOXX",
        )
        self.assertEqual(snapshot.snapshot_date, "2025-01-03")
        self.assertEqual(snapshot.nav, 10.0)
        self.assertEqual(snapshot.shares_outstanding, 1000)
        self.assertEqual(snapshot.total_net_assets, 10000.0)

        with self.assertRaisesRegex(
            ishares.ISharesDataValidationError,
            "dates do not match",
        ):
            ishares.parse_ishares_product_snapshot(
                product_html(assets_date="as of Jan 02, 2025"),
                requested_ticker="SOXX",
            )
        with self.assertRaises(ishares.ISharesTickerMismatchError):
            ishares.parse_ishares_product_snapshot(
                product_html("IGV"),
                requested_ticker="SOXX",
            )

    def test_official_aum_enriches_only_the_matching_date(self):
        history = ishares.parse_ishares_fund_history(
            spreadsheetml_bytes(),
            ticker="SOXX",
        )
        snapshot = ishares.parse_ishares_product_snapshot(
            product_html(),
            requested_ticker="SOXX",
        )
        enriched = ishares.enrich_history_with_official_snapshot(
            history,
            snapshot,
        )
        self.assertTrue(
            pd.isna(
                enriched.loc[
                    enriched["date"].eq("2025-01-02"),
                    "total_net_assets",
                ].iloc[0]
            )
        )
        self.assertEqual(
            enriched.loc[
                enriched["date"].eq("2025-01-03"),
                "total_net_assets",
            ].iloc[0],
            10000.0,
        )


class ISharesUpdateTests(unittest.TestCase):
    def test_update_writes_shared_csv_once_and_keeps_all_parsing_in_memory(self):
        config = load_sector_etf_config()
        soxx = next(etf for etf in config.ishares_etfs if etf.ticker == "SOXX")
        client = FakeHTTPClient(spreadsheetml_bytes(), product_html())
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            first = ishares.update_one_ishares_etf(
                soxx,
                config,
                output_dir=output_dir,
                http_client=client,
                max_attempts=1,
                sleep_func=lambda _: None,
            )
            output_path = output_dir / "soxx_semiconductors.csv"
            first_mtime = output_path.stat().st_mtime_ns
            second = ishares.update_one_ishares_etf(
                soxx,
                config,
                output_dir=output_dir,
                http_client=client,
                max_attempts=1,
                sleep_func=lambda _: None,
            )
            second_mtime = output_path.stat().st_mtime_ns
            files = sorted(path.name for path in output_dir.iterdir())
            saved = pd.read_csv(output_path)

        self.assertTrue(first.file_written)
        self.assertFalse(second.file_written)
        self.assertEqual(first_mtime, second_mtime)
        self.assertEqual(files, ["soxx_semiconductors.csv"])
        self.assertEqual(list(saved["date"]), ["2025-01-02", "2025-01-03"])
        self.assertFalse(saved["date"].duplicated().any())

    def test_invalid_download_preserves_existing_csv(self):
        config = load_sector_etf_config()
        soxx = next(etf for etf in config.ishares_etfs if etf.ticker == "SOXX")
        client = FakeHTTPClient(
            b"<html>Access Denied</html>",
            product_html(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "soxx_semiconductors.csv"
            original = (
                "date,nav,shares_outstanding,total_net_assets\n"
                "2025-01-02,9.5,900,8550\n"
            )
            output_path.write_text(original, encoding="utf-8")
            with self.assertRaises(ishares.ISharesFileFormatError):
                ishares.update_one_ishares_etf(
                    soxx,
                    config,
                    output_dir=tmp,
                    http_client=client,
                    max_attempts=1,
                    sleep_func=lambda _: None,
                )
            self.assertEqual(output_path.read_text(encoding="utf-8"), original)

    def test_one_ticker_failure_does_not_block_the_other(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                ishares,
                "update_one_ishares_etf",
                side_effect=[
                    RuntimeError("SOXX unavailable"),
                    ishares.FundHistoryUpdateResult(
                        ticker="IGV",
                        rows=2,
                        inserted_rows=2,
                        updated_rows=0,
                        file_written=True,
                        earliest_date="2025-01-02",
                        latest_date="2025-01-03",
                        consistency_warning_rows=0,
                        severe_consistency_rows=0,
                        zero_share_rows_normalized=0,
                    ),
                ],
            ),
        ):
            summary = ishares.run_ishares_etf_fund_history_update(
                output_dir=tmp,
            )
        self.assertEqual(summary.succeeded, 1)
        self.assertEqual(summary.failed, 1)
        self.assertIn("SOXX", summary.errors)


if __name__ == "__main__":
    unittest.main()

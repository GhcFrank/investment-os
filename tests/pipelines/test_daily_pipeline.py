import unittest
from unittest.mock import patch

from market_data import update_sector_etf_prices as sector_prices
from market_data.update_sector_etf_fund_history import (
    SectorETFFundHistorySummary,
)
from market_data.update_sector_etf_prices import (
    SectorETFPriceUpdateSummary,
)
from pipelines import run_daily_pipeline as pipeline


class DailyPipelineSectorETFTests(unittest.TestCase):
    def test_provider_steps_run_in_process_and_partial_fund_failure_continues(self):
        events = []
        price_summary = SectorETFPriceUpdateSummary(
            configured_etfs=11,
            price_tickers_succeeded=11,
        )
        fund_summary = SectorETFFundHistorySummary(
            configured_etfs=11,
            requested_etfs=11,
            succeeded=10,
            failed=1,
            files_written=2,
            files_unchanged=8,
            rows_inserted=2,
            rows_updated=1,
            consistency_warning_rows=0,
            severe_consistency_rows=0,
            zero_share_rows_normalized=0,
            mode="daily",
            errors={"XLY": "temporary failure"},
        )

        def record_script(path):
            events.append(path.name)

        def record_price_update():
            events.append("yahoo_sector_prices")
            return price_summary

        def record_fund_update():
            events.append("state_street_fund_history")
            return fund_summary

        with (
            patch.object(pipeline, "run_script", side_effect=record_script),
            patch.object(
                pipeline,
                "run_sector_etf_price_update",
                side_effect=record_price_update,
            ),
            patch.object(
                pipeline,
                "run_sector_etf_fund_history_update",
                side_effect=record_fund_update,
            ),
            patch.object(pipeline, "send_email"),
            patch.object(
                pipeline,
                "build_sentiment_email_section",
                return_value="sentiment",
            ),
        ):
            pipeline.main()

        self.assertEqual(
            events[:4],
            [
                "update_prices.py",
                "yahoo_sector_prices",
                "state_street_fund_history",
                "update_sentiment_indicators.py",
            ],
        )
        self.assertIn("daily_market_monitor.py", events)

    def test_yahoo_price_module_has_no_aum_fetcher(self):
        self.assertFalse(hasattr(sector_prices, "fetch_sector_etf_aum"))
        self.assertFalse(hasattr(sector_prices, "upsert_sector_etf_aum"))


if __name__ == "__main__":
    unittest.main()

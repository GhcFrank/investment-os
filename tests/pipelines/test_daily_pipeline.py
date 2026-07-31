import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

from market_data import update_sector_etf_prices as sector_prices
from market_data.update_sector_etf_fund_history import (
    SectorETFFundHistorySummary,
)
from market_data.update_ishares_etf_fund_history import (
    ISharesETFFundHistorySummary,
)
from market_data.update_sector_etf_prices import (
    SectorETFPriceUpdateSummary,
)
from pipelines import run_daily_pipeline as pipeline
from signals.build_sector_etf_metrics import (
    SectorETFMetricsUpdateSummary,
)


class DailyPipelineSectorETFTests(unittest.TestCase):
    def test_strict_sector_order_and_partial_fund_failure_continues(self):
        events = []
        price_summary = SectorETFPriceUpdateSummary(
            configured_etfs=13,
            price_tickers_succeeded=13,
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
        ishares_summary = ISharesETFFundHistorySummary(
            configured_etfs=2,
            requested_etfs=2,
            succeeded=1,
            failed=1,
            files_written=1,
            files_unchanged=0,
            rows_inserted=1,
            rows_updated=0,
            errors={"IGV": "temporary failure"},
        )
        metrics_summary = SectorETFMetricsUpdateSummary(
            configured_etfs=13,
            succeeded=13,
            failed=0,
            files_written=1,
            files_unchanged=12,
        )
        ranking_summary = MagicMock(
            email_status="success",
            email_error="",
        )
        ranking_summary.format.return_value = "ranking summary"

        def record_script(path):
            events.append(path.name)

        def record_price_update():
            events.append("yahoo_sector_prices")
            return price_summary

        def record_fund_update():
            events.append("state_street_fund_history")
            return fund_summary

        def record_ishares_update():
            events.append("ishares_fund_history")
            return ishares_summary

        def record_metrics_update():
            events.append("sector_etf_metrics")
            return metrics_summary

        def record_ranking_update(*, send_email_message):
            self.assertTrue(send_email_message)
            events.append("sector_etf_rankings_email")
            return ranking_summary

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
            patch.object(
                pipeline,
                "run_ishares_etf_fund_history_update",
                side_effect=record_ishares_update,
            ),
            patch.object(
                pipeline,
                "run_sector_etf_metrics_update",
                side_effect=record_metrics_update,
            ),
            patch.object(
                pipeline,
                "run_sector_etf_daily_ranking",
                side_effect=record_ranking_update,
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
            events[:7],
            [
                "update_prices.py",
                "state_street_fund_history",
                "ishares_fund_history",
                "yahoo_sector_prices",
                "sector_etf_metrics",
                "sector_etf_rankings_email",
                "update_sentiment_indicators.py",
            ],
        )
        self.assertIn("daily_market_monitor.py", events)

    def test_yahoo_price_module_has_no_aum_fetcher(self):
        self.assertFalse(hasattr(sector_prices, "fetch_sector_etf_aum"))
        self.assertFalse(hasattr(sector_prices, "upsert_sector_etf_aum"))

    def test_metrics_structural_failure_propagates(self):
        with patch.object(
            pipeline,
            "run_sector_etf_metrics_update",
            side_effect=ValueError("missing adj_close"),
        ):
            with self.assertRaisesRegex(ValueError, "missing adj_close"):
                pipeline.run_sector_etf_metrics_step()

    def test_metrics_step_logs_trading_day_horizons(self):
        complete = SectorETFMetricsUpdateSummary(
            configured_etfs=13,
            succeeded=13,
            failed=0,
            files_written=0,
            files_unchanged=13,
        )
        output = io.StringIO()
        with (
            patch.object(
                pipeline,
                "run_sector_etf_metrics_update",
                return_value=complete,
            ),
            redirect_stdout(output),
        ):
            pipeline.run_sector_etf_metrics_step()
        self.assertIn(
            "ETF return horizons: 30/90/250 trading days",
            output.getvalue(),
        )

    def test_incomplete_metrics_block_rankings(self):
        incomplete = SectorETFMetricsUpdateSummary(
            configured_etfs=13,
            succeeded=12,
            failed=1,
            files_written=12,
            files_unchanged=0,
            errors={"XLF": "write failed"},
        )
        with patch.object(
            pipeline,
            "run_sector_etf_metrics_update",
            return_value=incomplete,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "rankings and email are blocked",
            ):
                pipeline.run_sector_etf_metrics_step()

    def test_yahoo_structural_failure_stops_metrics_and_ranking(self):
        fund_summary = MagicMock()
        fund_summary.format.return_value = "fund summary"
        with (
            patch.object(pipeline, "run_script"),
            patch.object(
                pipeline,
                "run_sector_etf_fund_history_update",
                return_value=fund_summary,
            ),
            patch.object(
                pipeline,
                "run_ishares_etf_fund_history_update",
                return_value=MagicMock(
                    failed=0,
                    format=lambda: "ishares summary",
                ),
            ),
            patch.object(
                pipeline,
                "run_sector_etf_price_update",
                side_effect=RuntimeError("Yahoo structural failure"),
            ),
            patch.object(
                pipeline,
                "run_sector_etf_metrics_update",
            ) as metrics_update,
            patch.object(
                pipeline,
                "run_sector_etf_daily_ranking",
            ) as ranking_update,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Yahoo structural failure",
            ):
                pipeline.main()
        metrics_update.assert_not_called()
        ranking_update.assert_not_called()

    def test_metrics_failure_stops_ranking_email(self):
        fund_summary = MagicMock()
        fund_summary.format.return_value = "fund summary"
        price_summary = MagicMock()
        price_summary.format.return_value = "price summary"
        with (
            patch.object(pipeline, "run_script"),
            patch.object(
                pipeline,
                "run_sector_etf_fund_history_update",
                return_value=fund_summary,
            ),
            patch.object(
                pipeline,
                "run_ishares_etf_fund_history_update",
                return_value=MagicMock(
                    failed=0,
                    format=lambda: "ishares summary",
                ),
            ),
            patch.object(
                pipeline,
                "run_sector_etf_price_update",
                return_value=price_summary,
            ),
            patch.object(
                pipeline,
                "run_sector_etf_metrics_update",
                side_effect=ValueError("metrics invalid"),
            ),
            patch.object(
                pipeline,
                "run_sector_etf_daily_ranking",
            ) as ranking_update,
        ):
            with self.assertRaisesRegex(ValueError, "metrics invalid"):
                pipeline.main()
        ranking_update.assert_not_called()

    def test_ranking_email_failure_is_reported_after_local_save(self):
        summary = MagicMock(
            email_status="error",
            email_error="SMTP unavailable",
            history_written=True,
        )
        summary.format.return_value = "ranking saved; email failed"
        with patch.object(
            pipeline,
            "run_sector_etf_daily_ranking",
            return_value=summary,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "rankings were saved",
            ):
                pipeline.run_sector_etf_rankings_step()
        self.assertTrue(summary.history_written)


if __name__ == "__main__":
    unittest.main()

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
from market_data.gpu_cloud_summary import GPUCloudEmailSection
from market_data.gpu_cloud_status import API_KEY_MISSING
from market_data.vast_ai_client import VastAIAuthenticationError
from market_data.vix_sentiment_summary import (
    VIXMarketSentimentEmailSection,
)
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
            email_status="not_requested",
            email_error="",
        )
        ranking_summary.format.return_value = "ranking summary"
        ranking_summary.email = MagicMock(
            plain_text="rankings",
            html="<html><body><section>rankings</section></body></html>",
        )

        def record_script(path):
            events.append(path.name)

        def record_price_update():
            events.append("yahoo_sector_prices")
            return price_summary

        def record_gpu_update():
            events.append("vast_ai_search_offers")
            return MagicMock(format=lambda: "gpu market summary")

        def record_gpu_signals_update():
            events.append("gpu_cloud_signals")
            return MagicMock(format=lambda: "gpu signals summary")

        def record_vix_update():
            events.append("vix_market_data")
            return MagicMock(
                available=True,
                warnings=(),
                format=lambda: "vix update summary",
            )

        def record_vix_signals_update():
            events.append("vix_market_sentiment_signal")
            return MagicMock(
                data_status="SUCCESS",
                format=lambda: "vix signal summary",
            )

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
            self.assertFalse(send_email_message)
            events.append("sector_etf_rankings")
            return ranking_summary

        send_email_mock = MagicMock()
        with (
            patch.object(pipeline, "run_script", side_effect=record_script),
            patch.object(
                pipeline,
                "run_vix_market_update",
                side_effect=record_vix_update,
            ),
            patch.object(
                pipeline,
                "run_vix_market_sentiment_signals_update",
                side_effect=record_vix_signals_update,
            ),
            patch.object(
                pipeline,
                "run_gpu_cloud_market_update",
                side_effect=record_gpu_update,
            ),
            patch.object(
                pipeline,
                "run_gpu_cloud_market_signals_update",
                side_effect=record_gpu_signals_update,
            ),
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
            patch.object(pipeline, "send_email", send_email_mock),
            patch.object(
                pipeline,
                "build_vix_market_sentiment_email_section",
                return_value=VIXMarketSentimentEmailSection(
                    plain_text="VIX MARKET SENTIMENT",
                    html="<section>VIX MARKET SENTIMENT</section>",
                    available=True,
                    status="SUCCESS",
                ),
            ),
        ):
            pipeline.main()

        self.assertEqual(
            events[:11],
            [
                "vix_market_data",
                "vix_market_sentiment_signal",
                "vast_ai_search_offers",
                "gpu_cloud_signals",
                "update_prices.py",
                "state_street_fund_history",
                "ishares_fund_history",
                "yahoo_sector_prices",
                "sector_etf_metrics",
                "sector_etf_rankings",
                "build_sector_strength.py",
            ],
        )
        self.assertNotIn("update_sentiment_indicators.py", events)
        self.assertFalse(any("cnn" in event.lower() for event in events))
        self.assertIn("daily_market_monitor.py", events)
        send_email_mock.assert_called_once()
        self.assertNotIn(
            "Generated files",
            send_email_mock.call_args.kwargs["body"],
        )
        self.assertNotIn(
            "Generated files",
            send_email_mock.call_args.kwargs["html_body"],
        )

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
                "run_vix_steps_best_effort",
                return_value=pipeline.VIXPipelineResult(
                    available=False,
                    status="DATA_UNAVAILABLE",
                ),
            ),
            patch.object(
                pipeline,
                "run_gpu_cloud_market_update",
                return_value=MagicMock(format=lambda: "gpu market summary"),
            ),
            patch.object(
                pipeline,
                "run_gpu_cloud_market_signals_update",
                return_value=MagicMock(format=lambda: "gpu signals summary"),
            ),
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
                "run_vix_steps_best_effort",
                return_value=pipeline.VIXPipelineResult(
                    available=False,
                    status="DATA_UNAVAILABLE",
                ),
            ),
            patch.object(
                pipeline,
                "run_gpu_cloud_market_update",
                return_value=MagicMock(format=lambda: "gpu market summary"),
            ),
            patch.object(
                pipeline,
                "run_gpu_cloud_market_signals_update",
                return_value=MagicMock(format=lambda: "gpu signals summary"),
            ),
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

    def test_ranking_is_rendered_without_sending_a_separate_email(self):
        summary = MagicMock(
            email_status="not_requested",
            email_error="",
            history_written=True,
        )
        summary.format.return_value = "ranking saved"
        with patch.object(
            pipeline,
            "run_sector_etf_daily_ranking",
            return_value=summary,
        ) as ranking:
            result = pipeline.run_sector_etf_rankings_step()
        self.assertIs(result, summary)
        self.assertTrue(summary.history_written)
        ranking.assert_called_once_with(send_email_message=False)

    def test_gpu_market_collection_precedes_gpu_signals(self):
        events = []
        with (
            patch.object(
                pipeline,
                "run_gpu_cloud_market_update",
                side_effect=lambda: events.append("collection")
                or MagicMock(format=lambda: "collection"),
            ),
            patch.object(
                pipeline,
                "run_gpu_cloud_market_signals_update",
                side_effect=lambda: events.append("signals")
                or MagicMock(format=lambda: "signals"),
            ),
        ):
            pipeline.run_gpu_cloud_market_step()
            pipeline.run_gpu_cloud_market_signals_step()
        self.assertEqual(events, ["collection", "signals"])

    def test_gpu_warning_continues_to_signal_build(self):
        summary = MagicMock(
            warnings=("classification warning",),
            pricing_types_failed=0,
        )
        summary.format.return_value = "collection"
        signals = MagicMock()
        signals.format.return_value = "signals"
        with (
            patch.object(
                pipeline,
                "run_gpu_cloud_market_update",
                return_value=summary,
            ),
            patch.object(
                pipeline,
                "run_gpu_cloud_market_signals_update",
                return_value=signals,
            ) as signal_builder,
        ):
            result = pipeline.run_gpu_cloud_steps_best_effort()
        self.assertTrue(result.available)
        signal_builder.assert_called_once_with()

    def test_gpu_failure_continues_pipeline_and_one_unavailable_email(self):
        ranking = MagicMock()
        ranking.email = MagicMock(
            plain_text="rankings",
            html="<html><body><p>rankings</p></body></html>",
        )
        send_email_mock = MagicMock()
        unavailable = pipeline.GPUCloudPipelineResult(
            available=False,
            status=API_KEY_MISSING,
        )
        with (
            patch.object(pipeline, "run_script"),
            patch.object(
                pipeline,
                "run_vix_steps_best_effort",
                return_value=pipeline.VIXPipelineResult(
                    available=False,
                    status="DATA_UNAVAILABLE",
                ),
            ),
            patch.object(
                pipeline,
                "run_gpu_cloud_steps_best_effort",
                return_value=unavailable,
            ),
            patch.object(
                pipeline,
                "run_sector_etf_fund_history_step",
            ),
            patch.object(
                pipeline,
                "run_ishares_etf_fund_history_step",
            ),
            patch.object(pipeline, "run_sector_etf_price_step"),
            patch.object(pipeline, "run_sector_etf_metrics_step"),
            patch.object(
                pipeline,
                "run_sector_etf_rankings_step",
                return_value=ranking,
            ),
            patch.object(pipeline, "send_email", send_email_mock),
        ):
            pipeline.main()
        send_email_mock.assert_called_once()
        body = send_email_mock.call_args.kwargs["body"]
        self.assertIn("GPU market data unavailable", body)
        self.assertIn(API_KEY_MISSING, body)
        self.assertNotIn("Visible GPUs | 0", body)

    def test_missing_gpu_key_maps_to_safe_status_without_signals(self):
        with (
            patch.object(
                pipeline,
                "run_gpu_cloud_market_step",
                side_effect=VastAIAuthenticationError("secret-safe"),
            ),
            patch.object(
                pipeline,
                "run_gpu_cloud_market_signals_step",
            ) as signals,
        ):
            result = pipeline.run_gpu_cloud_steps_best_effort()
        self.assertFalse(result.available)
        self.assertEqual(result.status, API_KEY_MISSING)
        signals.assert_not_called()

    def test_daily_email_has_gpu_and_no_generated_files_in_both_formats(self):
        vix = VIXMarketSentimentEmailSection(
            plain_text="VIX MARKET SENTIMENT\nVIX level: 16.46",
            html="<section>VIX MARKET SENTIMENT VIX level: 16.46</section>",
            available=True,
            status="SUCCESS",
        )
        gpu = GPUCloudEmailSection(
            plain_text="GPU CLOUD SUPPLY — VAST.AI\nSupply Signal",
            html="<section>GPU CLOUD SUPPLY — VAST.AI Supply Signal</section>",
            available=True,
            status="SUCCESS",
        )
        plain, html = pipeline.build_daily_pipeline_email(
            run_time="2026-07-31 15:00:00",
            vix_section=vix,
            gpu_section=gpu,
        )
        self.assertIn("VIX MARKET SENTIMENT", plain)
        self.assertIn("VIX MARKET SENTIMENT", html)
        self.assertIn("GPU CLOUD SUPPLY", plain)
        self.assertIn("GPU CLOUD SUPPLY", html)
        self.assertNotIn("Generated Files", plain)
        self.assertNotIn("Generated files", plain)
        self.assertNotIn("Generated Files", html)
        self.assertNotIn("Generated files", html)
        self.assertNotIn("CNN", plain)
        self.assertNotIn("CNN", html)

    def test_vix_warning_continues_to_signal_build(self):
        update = MagicMock(
            available=True,
            warnings=("insufficient history",),
        )
        update.format.return_value = "vix update"
        signals = MagicMock(data_status="INSUFFICIENT_HISTORY")
        signals.format.return_value = "vix signals"
        with (
            patch.object(
                pipeline,
                "run_vix_market_update",
                return_value=update,
            ),
            patch.object(
                pipeline,
                "run_vix_market_sentiment_signals_update",
                return_value=signals,
            ) as signal_builder,
        ):
            result = pipeline.run_vix_steps_best_effort()
        self.assertTrue(result.available)
        self.assertEqual(result.status, "INSUFFICIENT_HISTORY")
        signal_builder.assert_called_once_with()

    def test_vix_failure_skips_signal_and_does_not_raise(self):
        update = MagicMock(available=False, warnings=())
        update.format.return_value = "vix unavailable"
        with (
            patch.object(
                pipeline,
                "run_vix_market_update",
                return_value=update,
            ),
            patch.object(
                pipeline,
                "run_vix_market_sentiment_signals_update",
            ) as signals,
        ):
            result = pipeline.run_vix_steps_best_effort()
        self.assertFalse(result.available)
        self.assertEqual(result.status, "DATA_UNAVAILABLE")
        signals.assert_not_called()

    def test_vix_failure_still_sends_one_daily_email_with_gpu(self):
        ranking = MagicMock()
        ranking.email = MagicMock(
            plain_text="rankings",
            html="<html><body><p>rankings</p></body></html>",
        )
        gpu_section = GPUCloudEmailSection(
            plain_text="GPU CLOUD SUPPLY — VAST.AI",
            html="<section>GPU CLOUD SUPPLY — VAST.AI</section>",
            available=True,
            status="SUCCESS",
        )
        send_email_mock = MagicMock()
        with (
            patch.object(pipeline, "run_script"),
            patch.object(
                pipeline,
                "run_vix_steps_best_effort",
                return_value=pipeline.VIXPipelineResult(
                    available=False,
                    status="DATA_UNAVAILABLE",
                ),
            ),
            patch.object(
                pipeline,
                "run_gpu_cloud_steps_best_effort",
                return_value=pipeline.GPUCloudPipelineResult(
                    available=True,
                    status="SUCCESS",
                ),
            ),
            patch.object(pipeline, "run_sector_etf_fund_history_step"),
            patch.object(pipeline, "run_ishares_etf_fund_history_step"),
            patch.object(pipeline, "run_sector_etf_price_step"),
            patch.object(pipeline, "run_sector_etf_metrics_step"),
            patch.object(
                pipeline,
                "run_sector_etf_rankings_step",
                return_value=ranking,
            ),
            patch.object(
                pipeline,
                "build_gpu_cloud_email_section",
                return_value=gpu_section,
            ),
            patch.object(pipeline, "send_email", send_email_mock),
        ):
            pipeline.main()
        send_email_mock.assert_called_once()
        body = send_email_mock.call_args.kwargs["body"]
        self.assertIn("VIX market sentiment data unavailable", body)
        self.assertIn("GPU CLOUD SUPPLY", body)
        self.assertNotIn("CNN", body)


if __name__ == "__main__":
    unittest.main()

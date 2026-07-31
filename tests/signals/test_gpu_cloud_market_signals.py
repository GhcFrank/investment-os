import tempfile
import unittest
from pathlib import Path

import pandas as pd

from market_data.gpu_cloud_config import (
    AVAILABILITY_SCOPE,
    GPU_CLOUD_FETCH_LOG_COLUMNS,
    GPU_CLOUD_HISTORY_COLUMNS,
    GPU_CLOUD_SIGNAL_COLUMNS,
    TRACKED_GPU_MODELS,
    VAST_SEARCH_OFFERS_ENDPOINT,
)
from market_data.gpu_cloud_status import DAILY_SELECTION_METHOD
from signals.build_gpu_cloud_market_signals import (
    build_gpu_cloud_market_signals,
    classify_supply_signal,
    run_gpu_cloud_market_signals_update,
)


def history_row(
    timestamp,
    offer_id,
    *,
    model="H100_SXM",
    pricing_type="on_demand",
    price=2.0,
    gpu_count=1,
    inventory_verifiable=True,
):
    return {
        "snapshot_timestamp_utc": timestamp,
        "snapshot_date": timestamp[:10],
        "provider": "vast_ai",
        "offer_id": offer_id,
        "machine_id": offer_id + 100,
        "host_id": offer_id + 200,
        "region": "New York, US",
        "country": "US",
        "gpu_name_raw": model.replace("_", " "),
        "gpu_model": model,
        "gpu_count": gpu_count,
        "pricing_type": pricing_type,
        "instance_price_per_hour_usd": (
            None if price is None or gpu_count is None else price * gpu_count
        ),
        "price_per_gpu_hour_usd": price,
        "is_available": True,
        "is_rentable": True,
        "verified": True,
        "reliability": 0.99,
        "min_bid_price_per_hour_usd": 0.8,
        "interruptible_price_per_gpu_hour_usd": (
            price if pricing_type == "interruptible" else None
        ),
        "inventory_count_is_verifiable": inventory_verifiable,
        "source_endpoint": VAST_SEARCH_OFFERS_ENDPOINT,
        "ingested_at_utc": timestamp,
    }


def fetch_row(
    timestamp,
    pricing_type,
    *,
    status="SUCCESS",
    notes="",
    offer_count=None,
    request_count=1,
    results_truncated=False,
):
    if offer_count is None:
        offer_count = (
            0
            if status
            in {
                "API_KEY_MISSING",
                "PROVIDER_ERROR",
                "SCHEMA_ERROR",
                "NO_MARKET_DATA",
            }
            else 1
        )
    return {
        "snapshot_timestamp_utc": timestamp,
        "snapshot_date": timestamp[:10],
        "provider": "vast_ai",
        "pricing_type": pricing_type,
        "status": status,
        "offer_count": offer_count,
        "request_count": request_count,
        "results_truncated": results_truncated,
        "data_quality_notes": notes,
        "source_endpoint": VAST_SEARCH_OFFERS_ENDPOINT,
        "ingested_at_utc": timestamp,
    }


def frames(history_rows, log_rows):
    return (
        pd.DataFrame(history_rows, columns=GPU_CLOUD_HISTORY_COLUMNS),
        pd.DataFrame(log_rows, columns=GPU_CLOUD_FETCH_LOG_COLUMNS),
    )


class GPUCloudMarketSignalsTests(unittest.TestCase):
    def test_first_run_metrics_counts_discount_and_quality(self):
        timestamp = "2026-07-31T12:00:00Z"
        history, log = frames(
            [
                history_row(timestamp, 1, price=1.0, gpu_count=2),
                history_row(timestamp, 2, price=2.0, gpu_count=1),
                history_row(timestamp, 3, price=3.0, gpu_count=4),
                history_row(
                    timestamp,
                    4,
                    pricing_type="interruptible",
                    price=1.0,
                    gpu_count=2,
                ),
            ],
            [fetch_row(timestamp, "on_demand"), fetch_row(timestamp, "interruptible")],
        )
        signals = build_gpu_cloud_market_signals(history, log)
        self.assertEqual(list(signals.columns), GPU_CLOUD_SIGNAL_COLUMNS)
        self.assertEqual(len(signals), len(TRACKED_GPU_MODELS))
        h100 = signals.loc[signals["gpu_model"].eq("H100_SXM")].iloc[0]
        self.assertEqual(h100["on_demand_median_price_per_gpu_hour"], 2.0)
        self.assertEqual(h100["on_demand_p25_price_per_gpu_hour"], 1.5)
        self.assertAlmostEqual(
            h100["on_demand_p10_price_per_gpu_hour"], 1.2
        )
        self.assertEqual(h100["visible_offer_count"], 3)
        self.assertEqual(h100["visible_gpu_count"], 7)
        self.assertTrue(pd.isna(h100["visible_offer_count_trend_7d"]))
        self.assertTrue(pd.isna(h100["visible_offer_count_trend_30d"]))
        self.assertEqual(h100["supply_signal"], "INSUFFICIENT_HISTORY")
        self.assertEqual(h100["interruptible_discount"], 0.5)
        self.assertTrue(h100["provider_available"])
        self.assertEqual(h100["cross_provider_availability"], 1.0)
        self.assertIsInstance(h100["cross_provider_availability"], float)
        self.assertEqual(h100["configured_provider_count"], 1)
        self.assertEqual(h100["availability_scope"], AVAILABILITY_SCOPE)
        self.assertEqual(
            h100["source_snapshot_timestamp_utc"], timestamp
        )
        self.assertEqual(
            h100["daily_snapshot_selection_method"],
            DAILY_SELECTION_METHOD,
        )
        self.assertEqual(h100["data_quality_status"], "INSUFFICIENT_HISTORY")

        no_market = signals.loc[signals["gpu_model"].eq("B200")].iloc[0]
        self.assertEqual(no_market["visible_offer_count"], 0)
        self.assertEqual(no_market["visible_gpu_count"], 0)
        self.assertFalse(no_market["provider_available"])
        self.assertEqual(no_market["data_quality_status"], "NO_MARKET_DATA")

    def test_natural_day_trends_prefer_exact_then_nearest_within_tolerance(self):
        dates_and_prices = [
            ("2026-06-29T12:00:00Z", 0.5),
            ("2026-07-22T12:00:00Z", 0.8),
            ("2026-07-23T12:00:00Z", 1.0),
            ("2026-07-30T12:00:00Z", 2.0),
        ]
        history_rows = []
        log_rows = []
        for index, (timestamp, price) in enumerate(dates_and_prices, start=1):
            history_rows.extend(
                [
                    history_row(timestamp, index, price=price),
                    history_row(
                        timestamp,
                        index + 100,
                        pricing_type="interruptible",
                        price=price / 2,
                    ),
                ]
            )
            log_rows.extend(
                [fetch_row(timestamp, "on_demand"), fetch_row(timestamp, "interruptible")]
            )
        history, log = frames(history_rows, log_rows)
        signals = build_gpu_cloud_market_signals(history, log)
        current = signals.loc[
            signals["date"].eq("2026-07-30")
            & signals["gpu_model"].eq("H100_SXM")
        ].iloc[0]
        self.assertEqual(current["rental_price_trend_7d"], 1.0)
        self.assertEqual(current["rental_price_trend_30d"], 3.0)
        self.assertEqual(current["visible_offer_count_trend_7d"], 0.0)
        self.assertEqual(current["visible_offer_count_trend_30d"], 0.0)
        self.assertIn("7d trend reference date: 2026-07-23", current["data_quality_notes"])
        self.assertIn("30d trend reference date: 2026-06-29", current["data_quality_notes"])
        self.assertEqual(current["data_quality_status"], "OK")

    def test_failed_request_is_unknown_not_zero_or_stale(self):
        success = "2026-07-30T12:00:00Z"
        failure = "2026-07-31T12:00:00Z"
        history, log = frames(
            [history_row(success, 1)],
            [
                fetch_row(success, "on_demand"),
                fetch_row(success, "interruptible"),
                fetch_row(
                    failure,
                    "on_demand",
                    status="PROVIDER_ERROR",
                    notes="provider_error",
                ),
                fetch_row(
                    failure,
                    "interruptible",
                    status="PROVIDER_ERROR",
                    notes="provider_error",
                ),
            ],
        )
        signals = build_gpu_cloud_market_signals(history, log)
        failed = signals.loc[
            signals["date"].eq("2026-07-31")
            & signals["gpu_model"].eq("H100_SXM")
        ].iloc[0]
        self.assertTrue(pd.isna(failed["provider_available"]))
        self.assertTrue(pd.isna(failed["visible_offer_count"]))
        self.assertTrue(pd.isna(failed["visible_gpu_count"]))
        self.assertTrue(pd.isna(failed["on_demand_median_price_per_gpu_hour"]))
        self.assertEqual(failed["providers_queried_successfully"], 0)
        self.assertEqual(failed["data_quality_status"], "PROVIDER_ERROR")
        self.assertEqual(failed["supply_signal"], "DATA_UNAVAILABLE")

    def test_partial_and_schema_warning_statuses_are_explicit(self):
        timestamp = "2026-07-31T12:00:00Z"
        history, log = frames(
            [
                history_row(
                    timestamp,
                    1,
                    price=2.0,
                    gpu_count=None,
                    inventory_verifiable=False,
                )
            ],
            [
                fetch_row(
                    timestamp,
                    "on_demand",
                    status="PARTIAL",
                    notes="schema_warning: invalid num_gpus",
                ),
                fetch_row(timestamp, "interruptible", status="PROVIDER_ERROR"),
            ],
        )
        signals = build_gpu_cloud_market_signals(history, log)
        row = signals.loc[signals["gpu_model"].eq("H100_SXM")].iloc[0]
        self.assertEqual(row["data_quality_status"], "SCHEMA_WARNING")
        self.assertEqual(row["visible_offer_count"], 1)
        self.assertTrue(pd.isna(row["visible_gpu_count"]))

    def test_latest_eligible_snapshot_wins_and_later_failure_is_ignored(self):
        timestamps = {
            "warning": "2026-07-31T14:32:00Z",
            "success": "2026-07-31T14:33:00Z",
            "failure": "2026-07-31T14:35:00Z",
            "latest": "2026-07-31T14:36:00Z",
        }
        history, log = frames(
            [
                history_row(timestamps["warning"], 1, price=1.0),
                history_row(timestamps["success"], 2, price=2.0),
                history_row(timestamps["latest"], 3, price=4.0),
                history_row(
                    timestamps["latest"],
                    4,
                    pricing_type="interruptible",
                    price=3.0,
                ),
            ],
            [
                fetch_row(
                    timestamps["warning"],
                    "on_demand",
                    status="SUCCESS_WITH_WARNINGS",
                    notes="gpu_model_warning: warning",
                ),
                fetch_row(timestamps["success"], "on_demand"),
                fetch_row(
                    timestamps["failure"],
                    "on_demand",
                    status="API_KEY_MISSING",
                    request_count=0,
                ),
                fetch_row(timestamps["latest"], "on_demand"),
                fetch_row(timestamps["latest"], "interruptible"),
            ],
        )
        signals = build_gpu_cloud_market_signals(history, log)
        row = signals.loc[signals["gpu_model"].eq("H100_SXM")].iloc[0]
        self.assertEqual(row["on_demand_median_price_per_gpu_hour"], 4.0)
        self.assertEqual(
            row["on_demand_source_snapshot_timestamp_utc"],
            timestamps["latest"],
        )
        self.assertNotIn("API_KEY_MISSING", row["data_quality_notes"])

        without_latest = log.loc[
            ~log["snapshot_timestamp_utc"].eq(timestamps["latest"])
        ]
        earlier_signals = build_gpu_cloud_market_signals(history, without_latest)
        earlier = earlier_signals.loc[
            earlier_signals["gpu_model"].eq("H100_SXM")
        ].iloc[0]
        self.assertEqual(earlier["on_demand_median_price_per_gpu_hour"], 2.0)
        self.assertEqual(
            earlier["on_demand_source_snapshot_timestamp_utc"],
            timestamps["success"],
        )

    def test_pricing_types_select_independent_timestamps(self):
        on_timestamp = "2026-07-31T14:33:00Z"
        bid_timestamp = "2026-07-31T14:36:00Z"
        history, log = frames(
            [
                history_row(on_timestamp, 1, price=2.0),
                history_row(
                    bid_timestamp,
                    2,
                    pricing_type="interruptible",
                    price=1.0,
                ),
            ],
            [
                fetch_row(on_timestamp, "on_demand"),
                fetch_row(bid_timestamp, "interruptible"),
            ],
        )
        signals = build_gpu_cloud_market_signals(history, log)
        row = signals.loc[signals["gpu_model"].eq("H100_SXM")].iloc[0]
        self.assertEqual(
            row["on_demand_source_snapshot_timestamp_utc"], on_timestamp
        )
        self.assertEqual(
            row["interruptible_source_snapshot_timestamp_utc"], bid_timestamp
        )
        self.assertEqual(row["interruptible_discount"], 0.5)

    def test_on_demand_success_and_interruptible_failure_is_partial_day(self):
        on_timestamp = "2026-07-31T14:33:00Z"
        bid_failure = "2026-07-31T14:36:00Z"
        history, log = frames(
            [history_row(on_timestamp, 1, price=2.0)],
            [
                fetch_row(on_timestamp, "on_demand"),
                fetch_row(
                    bid_failure,
                    "interruptible",
                    status="PROVIDER_ERROR",
                ),
            ],
        )
        signals = build_gpu_cloud_market_signals(history, log)
        row = signals.loc[signals["gpu_model"].eq("H100_SXM")].iloc[0]
        self.assertEqual(row["data_quality_status"], "PARTIAL_DAY")
        self.assertTrue(
            pd.isna(row["interruptible_source_snapshot_timestamp_utc"])
        )
        self.assertTrue(pd.isna(row["interruptible_discount"]))

    def test_legacy_partial_strict_compatibility_and_real_zero(self):
        eligible_partial = "2026-07-31T14:32:00Z"
        truncated_partial = "2026-07-31T14:33:00Z"
        history, log = frames(
            [history_row(eligible_partial, 1, price=2.0)],
            [
                fetch_row(
                    eligible_partial,
                    "on_demand",
                    status="PARTIAL",
                ),
                fetch_row(
                    truncated_partial,
                    "on_demand",
                    status="PARTIAL",
                    results_truncated=True,
                ),
                fetch_row(
                    truncated_partial,
                    "interruptible",
                    status="NO_MARKET_DATA",
                ),
            ],
        )
        signals = build_gpu_cloud_market_signals(history, log)
        row = signals.loc[signals["gpu_model"].eq("H100_SXM")].iloc[0]
        self.assertEqual(
            row["on_demand_source_snapshot_timestamp_utc"], eligible_partial
        )
        self.assertEqual(
            row["interruptible_source_snapshot_timestamp_utc"],
            truncated_partial,
        )
        self.assertIn("legacy_partial_compatibility", row["data_quality_notes"])

        zero_history, zero_log = frames(
            [],
            [
                fetch_row(
                    truncated_partial,
                    "on_demand",
                    status="NO_MARKET_DATA",
                ),
                fetch_row(
                    truncated_partial,
                    "interruptible",
                    status="NO_MARKET_DATA",
                ),
            ],
        )
        zero_signals = build_gpu_cloud_market_signals(zero_history, zero_log)
        zero = zero_signals.loc[
            zero_signals["gpu_model"].eq("H100_SXM")
        ].iloc[0]
        self.assertFalse(zero["provider_available"])
        self.assertEqual(zero["visible_offer_count"], 0)
        self.assertEqual(zero["cross_provider_availability"], 0.0)
        self.assertEqual(zero["data_quality_status"], "NO_MARKET_DATA")

    def test_offer_count_trends_use_natural_days_and_tolerance(self):
        snapshots = [
            ("2026-06-29T12:00:00Z", 2, 1.0),
            ("2026-07-23T12:00:00Z", 3, 1.5),
            ("2026-07-30T12:00:00Z", 4, 2.0),
        ]
        history_rows = []
        log_rows = []
        offer_id = 1
        for timestamp, offer_count, price in snapshots:
            for _ in range(offer_count):
                history_rows.append(
                    history_row(timestamp, offer_id, price=price)
                )
                offer_id += 1
            history_rows.append(
                history_row(
                    timestamp,
                    offer_id,
                    pricing_type="interruptible",
                    price=price / 2,
                )
            )
            offer_id += 1
            log_rows.extend(
                [
                    fetch_row(
                        timestamp,
                        "on_demand",
                        offer_count=offer_count,
                    ),
                    fetch_row(timestamp, "interruptible"),
                ]
            )
        history, log = frames(history_rows, log_rows)
        signals = build_gpu_cloud_market_signals(history, log)
        current = signals.loc[
            signals["date"].eq("2026-07-30")
            & signals["gpu_model"].eq("H100_SXM")
        ].iloc[0]
        self.assertAlmostEqual(
            current["visible_offer_count_trend_7d"], 4 / 3 - 1
        )
        self.assertEqual(current["visible_offer_count_trend_30d"], 1.0)
        self.assertIn(
            "30d offer-count trend reference date: 2026-06-29",
            current["data_quality_notes"],
        )

    def test_offer_count_zero_denominator_and_insufficient_history_are_blank(self):
        reference = "2026-07-23T12:00:00Z"
        current = "2026-07-30T12:00:00Z"
        history, log = frames(
            [history_row(current, 1)],
            [
                fetch_row(reference, "on_demand", status="NO_MARKET_DATA"),
                fetch_row(reference, "interruptible", status="NO_MARKET_DATA"),
                fetch_row(current, "on_demand"),
                fetch_row(current, "interruptible"),
            ],
        )
        signals = build_gpu_cloud_market_signals(history, log)
        row = signals.loc[
            signals["date"].eq("2026-07-30")
            & signals["gpu_model"].eq("H100_SXM")
        ].iloc[0]
        self.assertTrue(pd.isna(row["visible_offer_count_trend_7d"]))
        self.assertTrue(pd.isna(row["visible_offer_count_trend_30d"]))
        self.assertEqual(row["supply_signal"], "INSUFFICIENT_HISTORY")

    def test_offer_trend_uses_latest_eligible_snapshot_not_failure(self):
        reference = "2026-07-23T12:00:00Z"
        early = "2026-07-30T12:00:00Z"
        latest = "2026-07-30T13:00:00Z"
        failure = "2026-07-30T14:00:00Z"
        history_rows = [
            history_row(reference, 1),
            history_row(early, 2),
            history_row(early, 3),
            history_row(latest, 4),
            history_row(latest, 5),
            history_row(latest, 6),
        ]
        log_rows = [
            fetch_row(reference, "on_demand"),
            fetch_row(reference, "interruptible"),
            fetch_row(early, "on_demand", offer_count=2),
            fetch_row(latest, "on_demand", offer_count=3),
            fetch_row(latest, "interruptible"),
            fetch_row(failure, "on_demand", status="PROVIDER_ERROR"),
        ]
        history, log = frames(history_rows, log_rows)
        signals = build_gpu_cloud_market_signals(history, log)
        row = signals.loc[
            signals["date"].eq("2026-07-30")
            & signals["gpu_model"].eq("H100_SXM")
        ].iloc[0]
        self.assertEqual(row["visible_offer_count"], 3)
        self.assertEqual(row["visible_offer_count_trend_7d"], 2.0)
        self.assertEqual(
            row["on_demand_source_snapshot_timestamp_utc"], latest
        )

    def test_legacy_partial_can_supply_offer_trend_reference(self):
        reference = "2026-07-23T12:00:00Z"
        current = "2026-07-30T12:00:00Z"
        history, log = frames(
            [
                history_row(reference, 1),
                history_row(reference, 2),
                history_row(current, 3),
                history_row(current, 4),
                history_row(current, 5),
                history_row(current, 6),
            ],
            [
                fetch_row(
                    reference,
                    "on_demand",
                    status="PARTIAL",
                    offer_count=2,
                ),
                fetch_row(reference, "interruptible", status="PARTIAL"),
                fetch_row(current, "on_demand", offer_count=4),
                fetch_row(current, "interruptible"),
            ],
        )
        signals = build_gpu_cloud_market_signals(history, log)
        row = signals.loc[
            signals["date"].eq("2026-07-30")
            & signals["gpu_model"].eq("H100_SXM")
        ].iloc[0]
        self.assertEqual(row["visible_offer_count_trend_7d"], 1.0)
        self.assertIn("legacy_partial_compatibility", row["data_quality_notes"])

    def test_supply_signal_classifier_covers_all_statuses(self):
        cases = [
            ({"data_available": False}, "DATA_UNAVAILABLE"),
            ({}, "INSUFFICIENT_HISTORY"),
            (
                {
                    "rental_price_trend_7d": -0.01,
                    "rental_price_trend_30d": -0.11,
                    "visible_offer_count_trend_7d": 0.01,
                    "visible_offer_count_trend_30d": 0.21,
                },
                "OVERSUPPLY_WARNING",
            ),
            (
                {
                    "rental_price_trend_30d": -0.06,
                    "visible_offer_count_trend_30d": 0.15,
                },
                "LOOSENING",
            ),
            (
                {
                    "rental_price_trend_30d": 0.06,
                    "visible_offer_count_trend_30d": -0.15,
                },
                "TIGHTENING",
            ),
            (
                {
                    "rental_price_trend_30d": -0.01,
                    "visible_offer_count_trend_30d": 0.02,
                },
                "STABLE",
            ),
            (
                {
                    "rental_price_trend_30d": 0.10,
                    "visible_offer_count_trend_30d": 0.20,
                },
                "MIXED",
            ),
        ]
        defaults = {
            "rental_price_trend_7d": None,
            "rental_price_trend_30d": None,
            "visible_offer_count_trend_7d": None,
            "visible_offer_count_trend_30d": None,
            "data_available": True,
        }
        for overrides, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    classify_supply_signal(**(defaults | overrides)),
                    expected,
                )

    def test_signal_file_write_is_idempotent(self):
        timestamp = "2026-07-31T12:00:00Z"
        history, log = frames(
            [history_row(timestamp, 1)],
            [fetch_row(timestamp, "on_demand"), fetch_row(timestamp, "interruptible")],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history_file = root / "history.csv"
            log_file = root / "log.csv"
            output_file = root / "signals.csv"
            history.to_csv(history_file, index=False)
            log.to_csv(log_file, index=False)
            first = run_gpu_cloud_market_signals_update(
                history_file=history_file,
                fetch_log_file=log_file,
                output_file=output_file,
            )
            before = output_file.read_bytes()
            second = run_gpu_cloud_market_signals_update(
                history_file=history_file,
                fetch_log_file=log_file,
                output_file=output_file,
            )
            self.assertTrue(first.file_written)
            self.assertFalse(second.file_written)
            self.assertEqual(before, output_file.read_bytes())


if __name__ == "__main__":
    unittest.main()

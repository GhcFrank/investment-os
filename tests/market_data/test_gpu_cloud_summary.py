import tempfile
import unittest
from pathlib import Path

import pandas as pd

from market_data.gpu_cloud_config import GPU_CLOUD_SIGNAL_COLUMNS
from market_data.gpu_cloud_summary import build_gpu_cloud_email_section


def signal_row(**overrides):
    row = {column: "" for column in GPU_CLOUD_SIGNAL_COLUMNS}
    row.update(
        {
            "date": "2026-07-31",
            "gpu_model": "H100_SXM",
            "provider": "vast_ai",
            "source_snapshot_timestamp_utc": "2026-07-31T14:36:00Z",
            "on_demand_source_snapshot_timestamp_utc": (
                "2026-07-31T14:36:00Z"
            ),
            "interruptible_source_snapshot_timestamp_utc": (
                "2026-07-31T14:36:00Z"
            ),
            "snapshot_status": "SUCCESS_WITH_WARNINGS",
            "on_demand_snapshot_status": "SUCCESS_WITH_WARNINGS",
            "interruptible_snapshot_status": "SUCCESS_WITH_WARNINGS",
            "daily_snapshot_selection_method": (
                "latest_eligible_snapshot_per_pricing_type"
            ),
            "on_demand_median_price_per_gpu_hour": 2.0,
            "rental_price_trend_7d": "",
            "rental_price_trend_30d": "",
            "visible_offer_count_trend_7d": "",
            "visible_offer_count_trend_30d": "",
            "visible_offer_count": 4,
            "visible_gpu_count": 8,
            "supply_signal": "INSUFFICIENT_HISTORY",
            "provider_available": True,
            "configured_provider_count": 1,
            "providers_queried_successfully": 1,
            "providers_available": 1,
            "cross_provider_availability": 1.0,
            "availability_scope": "Vast.ai only; not yet cross-provider",
            "inventory_scope": "visible market",
            "data_quality_status": "INSUFFICIENT_HISTORY",
            "data_quality_notes": (
                "gpu_model_warning: H200 form factor is not specified"
            ),
        }
    )
    row.update(overrides)
    return row


class GPUCloudSummaryTests(unittest.TestCase):
    def _render(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signals.csv"
            pd.DataFrame(rows, columns=GPU_CLOUD_SIGNAL_COLUMNS).to_csv(
                path,
                index=False,
            )
            return build_gpu_cloud_email_section(path)

    def test_section_contains_required_metrics_and_safe_missing_history(self):
        section = self._render([signal_row()])
        self.assertTrue(section.available)
        for label in (
            "GPU CLOUD SUPPLY — VAST.AI",
            "7D Rental Price",
            "30D Rental Price",
            "Visible GPUs",
            "7D Offer Trend",
            "30D Offer Trend",
            "Supply Signal",
            "Insufficient history",
            "INSUFFICIENT_HISTORY",
            "2026-07-31 14:36 UTC",
            "Vast.ai visible marketplace only",
        ):
            self.assertIn(label, section.plain_text)
        self.assertNotIn("nan", section.plain_text.lower())
        self.assertNotIn("None", section.plain_text)
        self.assertNotIn("Generated Files", section.plain_text)
        self.assertNotIn("Generated Files", section.html)

    def test_percent_format_and_on_demand_only_missing_price(self):
        section = self._render(
            [
                signal_row(
                    rental_price_trend_7d=0.126,
                    rental_price_trend_30d=-0.084,
                    visible_offer_count_trend_7d=0.20,
                    visible_offer_count_trend_30d=-0.10,
                    supply_signal="MIXED",
                ),
                signal_row(
                    gpu_model="H100_PCIE",
                    on_demand_median_price_per_gpu_hour="",
                    rental_price_trend_7d="",
                    rental_price_trend_30d="",
                    visible_gpu_count=0,
                ),
            ]
        )
        self.assertIn("+12.6%", section.plain_text)
        self.assertIn("-8.4%", section.plain_text)
        h100_pcie_line = next(
            line
            for line in section.plain_text.splitlines()
            if line.startswith("H100_PCIE")
        )
        self.assertIn("N/A", h100_pcie_line)

    def test_failed_snapshot_is_unavailable_not_zero_and_hides_raw_notes(self):
        section = self._render(
            [
                signal_row(
                    snapshot_status="API_KEY_MISSING",
                    on_demand_snapshot_status="API_KEY_MISSING",
                    source_snapshot_timestamp_utc="",
                    on_demand_median_price_per_gpu_hour="",
                    visible_offer_count="",
                    visible_gpu_count="",
                    supply_signal="DATA_UNAVAILABLE",
                    data_quality_status="API_KEY_MISSING",
                    data_quality_notes="VAST_API_KEY=do-not-display",
                )
            ]
        )
        self.assertFalse(section.available)
        self.assertIn("GPU market data unavailable", section.plain_text)
        self.assertIn("API_KEY_MISSING", section.plain_text)
        self.assertNotIn("Visible GPUs", section.plain_text)
        self.assertNotIn("do-not-display", section.plain_text)
        self.assertNotIn("do-not-display", section.html)

    def test_missing_or_legacy_schema_degrades_to_schema_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.csv"
            pd.DataFrame([{"date": "2026-07-31"}]).to_csv(
                path,
                index=False,
            )
            section = build_gpu_cloud_email_section(path)
        self.assertFalse(section.available)
        self.assertIn("SCHEMA_ERROR", section.plain_text)


if __name__ == "__main__":
    unittest.main()

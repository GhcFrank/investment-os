import tempfile
import unittest
from pathlib import Path

import pandas as pd

from market_data.vix_sentiment_config import (
    VIX_MARKET_SENTIMENT_SIGNAL_COLUMNS,
)
from market_data.vix_sentiment_summary import (
    build_vix_market_sentiment_email_section,
)


def signal_row(**overrides):
    row = {
        "date": "2026-07-31",
        "vix": 16.46,
        "change_1d": -0.63,
        "change_5d": -2.12,
        "change_20d": 0.65,
        "sentiment_regime": "NORMAL",
        "signal": "NEUTRAL",
        "source": "yfinance",
        "source_status": "ok",
        "data_status": "SUCCESS",
        "stale": False,
        "updated_at": "2026-07-31",
        "data_quality_notes": "",
    }
    row.update(overrides)
    return row


class VIXMarketSentimentSummaryTests(unittest.TestCase):
    def _render(self, row):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vix_signal.csv"
            pd.DataFrame(
                [row],
                columns=VIX_MARKET_SENTIMENT_SIGNAL_COLUMNS,
            ).to_csv(path, index=False)
            return build_vix_market_sentiment_email_section(path)

    def test_success_section_has_required_fields_in_plain_and_html(self):
        section = self._render(signal_row())
        self.assertTrue(section.available)
        for label in (
            "VIX MARKET SENTIMENT",
            "VIX level: 16.46",
            "Daily change: -0.63 points",
            "5-day change: -2.12 points",
            "20-day change: +0.65 points",
            "Current sentiment regime: NORMAL",
            "Signal / interpretation: NEUTRAL",
            "Data date: 2026-07-31",
            "Data status: SUCCESS",
        ):
            self.assertIn(label, section.plain_text)
        self.assertIn("VIX MARKET SENTIMENT", section.html)
        self.assertNotIn("CNN", section.plain_text)
        self.assertNotIn("Generated Files", section.plain_text)

    def test_insufficient_history_is_not_zero_nan_or_none(self):
        section = self._render(
            signal_row(
                change_5d="",
                change_20d="",
                data_status="INSUFFICIENT_HISTORY",
            )
        )
        self.assertIn("Insufficient history", section.plain_text)
        self.assertNotIn("nan", section.plain_text.lower())
        self.assertNotIn("None", section.plain_text)

    def test_failure_is_safe_unavailable_and_hides_notes(self):
        section = self._render(
            signal_row(
                vix="",
                data_status="DATA_UNAVAILABLE",
                data_quality_notes="API_KEY=do-not-display",
            )
        )
        self.assertFalse(section.available)
        self.assertIn(
            "VIX market sentiment data unavailable",
            section.plain_text,
        )
        self.assertNotIn("do-not-display", section.plain_text)
        self.assertNotIn("do-not-display", section.html)

    def test_stale_signal_is_disclosed(self):
        section = self._render(
            signal_row(
                date="2026-07-30",
                data_status="SUCCESS_WITH_WARNINGS",
                stale=True,
            )
        )
        self.assertIn("Data date: 2026-07-30", section.plain_text)
        self.assertIn("Stale: Yes", section.plain_text)


if __name__ == "__main__":
    unittest.main()

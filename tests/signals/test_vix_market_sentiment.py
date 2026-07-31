import tempfile
import unittest
from pathlib import Path

import pandas as pd

from market_data.update_sentiment_indicators import VIX_COLUMNS, vix_row
from market_data.vix_sentiment_config import (
    VIX_MARKET_SENTIMENT_SIGNAL_COLUMNS,
)
from signals.build_vix_market_sentiment import (
    build_vix_market_sentiment_signal,
    run_vix_market_sentiment_signals_update,
)


def current_frame(**overrides):
    row = vix_row(
        date="2026-07-31",
        value=16.46,
        level="normal",
        updated_at="2026-07-31",
    )
    row.update(
        {
            "change_1d": "-0.63",
            "change_5d": "-2.12",
            "change_20d": "0.65",
        }
    )
    row.update(overrides)
    return pd.DataFrame([row], columns=VIX_COLUMNS)


class VIXMarketSentimentSignalTests(unittest.TestCase):
    def test_success_signal_maps_existing_regime_and_point_changes(self):
        signal = build_vix_market_sentiment_signal(
            current_frame(),
            current_et_date="2026-07-31",
        )
        self.assertEqual(
            list(signal.columns),
            VIX_MARKET_SENTIMENT_SIGNAL_COLUMNS,
        )
        row = signal.iloc[-1]
        self.assertEqual(row["vix"], 16.46)
        self.assertEqual(row["change_1d"], -0.63)
        self.assertEqual(row["sentiment_regime"], "NORMAL")
        self.assertEqual(row["signal"], "NEUTRAL")
        self.assertEqual(row["data_status"], "SUCCESS")
        self.assertFalse(row["stale"])

    def test_missing_history_is_explicit_not_zero(self):
        signal = build_vix_market_sentiment_signal(
            current_frame(change_5d="", change_20d=""),
            current_et_date="2026-07-31",
        )
        row = signal.iloc[-1]
        self.assertEqual(row["data_status"], "INSUFFICIENT_HISTORY")
        self.assertTrue(pd.isna(row["change_5d"]))
        self.assertTrue(pd.isna(row["change_20d"]))

    def test_old_success_is_marked_stale_with_warning(self):
        signal = build_vix_market_sentiment_signal(
            current_frame(date="2026-07-30"),
            current_et_date="2026-07-31",
        )
        row = signal.iloc[-1]
        self.assertEqual(row["data_status"], "SUCCESS_WITH_WARNINGS")
        self.assertTrue(row["stale"])

    def test_failed_formal_observation_is_data_unavailable(self):
        signal = build_vix_market_sentiment_signal(
            current_frame(
                vix="",
                status="failed",
                change_1d="",
                change_5d="",
                change_20d="",
            ),
            current_et_date="2026-07-31",
        )
        self.assertEqual(
            signal.iloc[-1]["data_status"],
            "DATA_UNAVAILABLE",
        )

    def test_signal_file_write_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current_file = root / "vix.csv"
            output_file = root / "vix_signal.csv"
            current_frame().to_csv(current_file, index=False)
            first = run_vix_market_sentiment_signals_update(
                current_file=current_file,
                output_file=output_file,
                current_et_date="2026-07-31",
            )
            before = output_file.read_bytes()
            second = run_vix_market_sentiment_signals_update(
                current_file=current_file,
                output_file=output_file,
                current_et_date="2026-07-31",
            )
            self.assertTrue(first.file_written)
            self.assertFalse(second.file_written)
            self.assertEqual(before, output_file.read_bytes())


if __name__ == "__main__":
    unittest.main()

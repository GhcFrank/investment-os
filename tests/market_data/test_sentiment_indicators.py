import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import requests

from market_data import update_sentiment_indicators as sentiment
from market_data.sentiment_summary import build_sentiment_email_section


class SentimentLevelTests(unittest.TestCase):
    def test_vix_level_boundaries(self):
        self.assertEqual(sentiment.vix_level(14.9), "calm")
        self.assertEqual(sentiment.vix_level(15.0), "normal")
        self.assertEqual(sentiment.vix_level(19.9), "normal")
        self.assertEqual(sentiment.vix_level(20.0), "risk_off")
        self.assertEqual(sentiment.vix_level(29.9), "risk_off")
        self.assertEqual(sentiment.vix_level(30.0), "stress")

    def test_cnn_level_boundaries(self):
        self.assertEqual(sentiment.cnn_level(10), "extreme_fear")
        self.assertEqual(sentiment.cnn_level(30), "fear")
        self.assertEqual(sentiment.cnn_level(50), "neutral")
        self.assertEqual(sentiment.cnn_level(60), "greed")
        self.assertEqual(sentiment.cnn_level(80), "extreme_greed")


class FakeResponse:
    def __init__(self, payload=None, status_code=200, json_error=None):
        self.payload = payload
        self.status_code = status_code
        self.json_error = json_error

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        if self.json_error:
            raise self.json_error
        return self.payload


class CnnFetchTests(unittest.TestCase):
    def test_cnn_parse_current_score(self):
        row = sentiment.parse_cnn_payload(
            {
                "fear_and_greed": {
                    "score": 42,
                    "rating": "fear",
                    "timestamp": 1710000000000,
                }
            }
        )

        self.assertEqual(row["fear_greed_index"], "42.00")
        self.assertEqual(row["level"], "fear")
        self.assertEqual(row["status"], "ok")

    def test_cnn_failure_returns_failed_row(self):
        with patch.object(
            sentiment.requests,
            "get",
            side_effect=[
                FakeResponse(status_code=403),
                FakeResponse(json_error=ValueError("bad json")),
            ],
        ):
            row = sentiment.fetch_cnn_fear_greed()

        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["fear_greed_index"], "")
        self.assertEqual(
            row["error_message"],
            "CNN Fear & Greed unavailable: HTTP 403 from CNN graphdata endpoint",
        )

    def test_cnn_418_error_is_shortened(self):
        message = sentiment.summarize_cnn_error(
            "https://example.test/path: 418 Client Error: Unknown Error"
        )

        self.assertEqual(
            message,
            "CNN Fear & Greed unavailable: HTTP 418 from CNN graphdata endpoint",
        )
        self.assertLessEqual(len(message), 160)


class SentimentCsvTests(unittest.TestCase):
    def test_vix_output_schema(self):
        row = sentiment.vix_row(
            date="2026-06-25",
            value=18,
            level="normal",
        )
        rows = pd.DataFrame([row], columns=sentiment.VIX_COLUMNS)

        self.assertEqual(list(rows.columns), sentiment.VIX_COLUMNS)

    def test_cnn_output_schema(self):
        row = sentiment.cnn_row(
            date="2026-06-25",
            value=42,
            level="fear",
        )
        rows = pd.DataFrame([row], columns=sentiment.CNN_COLUMNS)

        self.assertEqual(list(rows.columns), sentiment.CNN_COLUMNS)

    def test_vix_history_upsert_replaces_date(self):
        existing = pd.DataFrame(
            [
                sentiment.vix_row(
                    date="2026-06-25",
                    value=18,
                    level="normal",
                )
            ],
            columns=sentiment.VIX_COLUMNS,
        )
        replacement = pd.DataFrame(
            [
                sentiment.vix_row(
                    date="2026-06-25",
                    value=19,
                    level="normal",
                )
            ],
            columns=sentiment.VIX_COLUMNS,
        )

        history = sentiment.upsert_history(
            existing,
            replacement,
            sentiment.VIX_COLUMNS,
        )

        self.assertEqual(len(history), 1)
        self.assertEqual(history.loc[0, "vix"], "19.00")

    def test_cnn_history_upsert_replaces_date(self):
        existing = pd.DataFrame(
            [
                sentiment.cnn_row(
                    date="2026-06-25",
                    value=41,
                    level="fear",
                )
            ],
            columns=sentiment.CNN_COLUMNS,
        )
        replacement = pd.DataFrame(
            [
                sentiment.cnn_row(
                    date="2026-06-25",
                    value=42,
                    level="fear",
                )
            ],
            columns=sentiment.CNN_COLUMNS,
        )

        history = sentiment.upsert_history(
            existing,
            replacement,
            sentiment.CNN_COLUMNS,
        )

        self.assertEqual(len(history), 1)
        self.assertEqual(history.loc[0, "fear_greed_index"], "42.00")

    def test_change_calculation_uses_observations(self):
        historical_rows = []
        for index in range(1, 22):
            historical_rows.append(
                sentiment.vix_row(
                    date=f"2026-06-{index:02d}",
                    value=10 + index,
                    level="normal",
                )
            )
        history = pd.DataFrame(historical_rows, columns=sentiment.VIX_COLUMNS)
        rows = pd.DataFrame(
            [
                sentiment.vix_row(
                    date="2026-06-22",
                    value=40,
                    level="stress",
                )
            ],
            columns=sentiment.VIX_COLUMNS,
        )

        changed = sentiment.add_change_columns(
            rows,
            history,
            "vix",
            sentiment.VIX_COLUMNS,
        )

        self.assertEqual(changed.loc[0, "change_1d"], "9.00")
        self.assertEqual(changed.loc[0, "change_5d"], "13.00")
        self.assertEqual(changed.loc[0, "change_20d"], "28.00")


class SentimentEmailSectionTests(unittest.TestCase):
    def test_email_section_reads_split_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            vix_file = tmp_path / "vix.csv"
            cnn_file = tmp_path / "cnn_fear_greed.csv"
            pd.DataFrame(
                [
                    sentiment.vix_row(
                        date="2026-06-25",
                        value=18.42,
                        level="normal",
                    )
                ],
                columns=sentiment.VIX_COLUMNS,
            ).to_csv(vix_file, index=False)
            pd.DataFrame(
                [
                    sentiment.cnn_row(
                        date="2026-06-25",
                        value=42,
                        level="fear",
                    )
                ],
                columns=sentiment.CNN_COLUMNS,
            ).to_csv(cnn_file, index=False)

            section = build_sentiment_email_section(vix_file, cnn_file)

        self.assertIn("Market Sentiment", section)
        self.assertIn("VIX: 18.42 (normal)", section)
        self.assertIn("CNN Fear & Greed: 42.00 (fear)", section)

    def test_email_section_includes_unavailable_cnn(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            vix_file = tmp_path / "vix.csv"
            cnn_file = tmp_path / "cnn_fear_greed.csv"
            pd.DataFrame(
                [
                    sentiment.vix_row(
                        date="2026-06-25",
                        value=18.42,
                        level="normal",
                    )
                ],
                columns=sentiment.VIX_COLUMNS,
            ).to_csv(vix_file, index=False)
            pd.DataFrame(
                [sentiment.failed_cnn_row("418 Client Error")],
                columns=sentiment.CNN_COLUMNS,
            ).to_csv(cnn_file, index=False)

            section = build_sentiment_email_section(vix_file, cnn_file)

        self.assertIn("CNN Fear & Greed: unavailable", section)
        self.assertIn("HTTP 418", section)


if __name__ == "__main__":
    unittest.main()

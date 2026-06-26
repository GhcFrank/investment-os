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

        self.assertEqual(row["indicator"], "CNN_FEAR_GREED")
        self.assertEqual(row["value"], "42.00")
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

        self.assertEqual(row["indicator"], "CNN_FEAR_GREED")
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["value"], "")
        self.assertIn("HTTP 403", row["error_message"])
        self.assertIn("bad json", row["error_message"])


class SentimentCsvTests(unittest.TestCase):
    def test_history_upsert_replaces_date_indicator(self):
        existing = pd.DataFrame(
            [
                sentiment.sentiment_row(
                    date="2026-06-25",
                    indicator="VIX",
                    value=18,
                    level="normal",
                    source="yfinance",
                    status="ok",
                )
            ],
            columns=sentiment.SENTIMENT_COLUMNS,
        )
        replacement = pd.DataFrame(
            [
                sentiment.sentiment_row(
                    date="2026-06-25",
                    indicator="VIX",
                    value=19,
                    level="normal",
                    source="yfinance",
                    status="ok",
                )
            ],
            columns=sentiment.SENTIMENT_COLUMNS,
        )

        history = sentiment.upsert_history(existing, replacement)

        self.assertEqual(len(history), 1)
        self.assertEqual(history.loc[0, "value"], "19.00")

    def test_change_calculation_uses_observations(self):
        historical_rows = []
        for index in range(1, 22):
            historical_rows.append(
                sentiment.sentiment_row(
                    date=f"2026-06-{index:02d}",
                    indicator="VIX",
                    value=10 + index,
                    level="normal",
                    source="yfinance",
                    status="ok",
                )
            )
        history = pd.DataFrame(historical_rows, columns=sentiment.SENTIMENT_COLUMNS)
        rows = pd.DataFrame(
            [
                sentiment.sentiment_row(
                    date="2026-06-22",
                    indicator="VIX",
                    value=40,
                    level="stress",
                    source="yfinance",
                    status="ok",
                )
            ],
            columns=sentiment.SENTIMENT_COLUMNS,
        )

        changed = sentiment.add_change_columns(rows, history)

        self.assertEqual(changed.loc[0, "change_1d"], "9.00")
        self.assertEqual(changed.loc[0, "change_5d"], "13.00")
        self.assertEqual(changed.loc[0, "change_20d"], "28.00")


class SentimentEmailSectionTests(unittest.TestCase):
    def test_email_section_includes_vix_and_cnn(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sentiment.csv"
            rows = pd.DataFrame(
                [
                    sentiment.sentiment_row(
                        date="2026-06-25",
                        indicator="VIX",
                        value=18.42,
                        level="normal",
                        source="yfinance",
                        status="ok",
                    ),
                    sentiment.sentiment_row(
                        date="2026-06-25",
                        indicator="CNN_FEAR_GREED",
                        value=42,
                        level="fear",
                        source="cnn_unofficial",
                        status="ok",
                    ),
                ],
                columns=sentiment.SENTIMENT_COLUMNS,
            )
            rows.to_csv(path, index=False)

            section = build_sentiment_email_section(path)

        self.assertIn("Market Sentiment", section)
        self.assertIn("VIX: 18.42 (normal)", section)
        self.assertIn("CNN Fear & Greed: 42.00 (fear)", section)

    def test_email_section_includes_unavailable_cnn(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sentiment.csv"
            rows = pd.DataFrame(
                [
                    sentiment.sentiment_row(
                        date="2026-06-25",
                        indicator="VIX",
                        value=18.42,
                        level="normal",
                        source="yfinance",
                        status="ok",
                    ),
                    sentiment.failed_cnn_row("HTTP 403"),
                ],
                columns=sentiment.SENTIMENT_COLUMNS,
            )
            rows.to_csv(path, index=False)

            section = build_sentiment_email_section(path)

        self.assertIn("CNN Fear & Greed: unavailable", section)
        self.assertIn("- status: failed", section)
        self.assertIn("- error: HTTP 403", section)


if __name__ == "__main__":
    unittest.main()

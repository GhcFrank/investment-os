import unittest
from datetime import UTC, date, datetime
from unittest.mock import patch

import requests

from news import news_utils


class FakeResponse:
    def __init__(self, status_code, headers=None):
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return []


class GetJsonRetryTests(unittest.TestCase):
    def test_to_yyyy_mm_dd_normalizes_date_like_values(self):
        self.assertEqual(
            news_utils.to_yyyy_mm_dd("2026-06-20T15:30:12Z"),
            "2026-06-20",
        )
        self.assertEqual(news_utils.to_yyyy_mm_dd("2026-06-20"), "2026-06-20")
        self.assertEqual(
            news_utils.to_yyyy_mm_dd("2026-06-20 15:30:12+00:00"),
            "2026-06-20",
        )
        self.assertEqual(
            news_utils.to_yyyy_mm_dd(datetime(2026, 6, 20, 15, 30, tzinfo=UTC)),
            "2026-06-20",
        )
        self.assertEqual(news_utils.to_yyyy_mm_dd(date(2026, 6, 20)), "2026-06-20")
        self.assertEqual(news_utils.to_yyyy_mm_dd(""), "")
        self.assertEqual(news_utils.to_yyyy_mm_dd(None), "")
        self.assertEqual(news_utils.to_yyyy_mm_dd("not a date"), "")

    def test_404_does_not_retry(self):
        with patch.object(news_utils.SESSION, "get", return_value=FakeResponse(404)) as get:
            with self.assertRaises(requests.HTTPError):
                news_utils.get_json("https://example.test/missing")

        self.assertEqual(get.call_count, 1)

    def test_503_retries(self):
        with patch.object(
            news_utils.SESSION,
            "get",
            side_effect=[FakeResponse(503), FakeResponse(200)],
        ) as get:
            with patch.object(news_utils.time, "sleep") as sleep:
                response = news_utils.get_json("https://example.test/busy")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(get.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_429_honors_retry_after(self):
        with patch.object(
            news_utils.SESSION,
            "get",
            side_effect=[FakeResponse(429, {"Retry-After": "3"}), FakeResponse(200)],
        ):
            with patch.object(news_utils.time, "sleep") as sleep:
                news_utils.get_json("https://example.test/rate-limited")

        sleep.assert_called_once_with(3.0)

    def test_timeout_retries(self):
        with patch.object(
            news_utils.SESSION,
            "get",
            side_effect=[requests.Timeout("timeout"), FakeResponse(200)],
        ) as get:
            with patch.object(news_utils.time, "sleep") as sleep:
                response = news_utils.get_json("https://example.test/slow")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(get.call_count, 2)
        sleep.assert_called_once_with(1)

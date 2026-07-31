import unittest
from datetime import UTC, datetime

import requests

from market_data.gpu_cloud_config import VAST_SEARCH_OFFERS_ENDPOINT
from market_data.vast_ai_client import (
    VastAIAuthenticationError,
    VastAIClient,
    VastAISchemaError,
    canonicalize_gpu_model,
    normalize_vast_offers,
)


class FakeResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def offer(offer_id, gpu_name="H100 SXM", **overrides):
    row = {
        "id": offer_id,
        "machine_id": 100 + offer_id,
        "host_id": 200 + offer_id,
        "geolocation": "New York, US",
        "gpu_name": gpu_name,
        "gpu_ram": 81920,
        "num_gpus": 2,
        "dph_total": 4.0,
        "min_bid": 2.0,
        "rentable": True,
        "rented": False,
        "verification": "verified",
        "reliability2": 0.99,
    }
    row.update(overrides)
    return row


class VastAIClientTests(unittest.TestCase):
    def test_search_uses_only_documented_read_endpoint_and_filters(self):
        session = FakeSession([FakeResponse(body={"offers": [offer(1)]})])
        client = VastAIClient("secret-value", session=session)

        result = client.search_offers(
            pricing_type="on_demand",
            gpu_names=["H100 SXM"],
        )

        self.assertEqual(len(result.offers), 1)
        self.assertEqual(result.request_count, 1)
        url, kwargs = session.calls[0]
        self.assertEqual(url, VAST_SEARCH_OFFERS_ENDPOINT)
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret-value")
        self.assertEqual(kwargs["json"]["type"], "ondemand")
        self.assertEqual(kwargs["json"]["gpu_name"], {"in": ["H100 SXM"]})
        self.assertEqual(kwargs["json"]["verified"], {"eq": True})
        self.assertEqual(kwargs["json"]["rentable"], {"eq": True})
        self.assertEqual(kwargs["json"]["rented"], {"eq": False})

    def test_rate_limit_retries_with_exponential_backoff(self):
        session = FakeSession(
            [
                FakeResponse(status_code=429, body={}),
                FakeResponse(status_code=503, body={}),
                FakeResponse(body={"offers": []}),
            ]
        )
        sleeps = []
        client = VastAIClient(
            "secret",
            session=session,
            max_attempts=3,
            retry_base_seconds=0.25,
            sleep_func=sleeps.append,
        )
        result = client.search_offers(
            pricing_type="interruptible",
            gpu_names=["L40S"],
        )
        self.assertEqual(result.offers, ())
        self.assertEqual(result.request_count, 3)
        self.assertEqual(sleeps, [0.25, 0.5])
        self.assertEqual(session.calls[-1][1]["json"]["type"], "bid")

    def test_timeout_error_never_contains_api_key(self):
        key = "must-not-leak"
        session = FakeSession([requests.Timeout("transport timeout")])
        client = VastAIClient(key, session=session, max_attempts=1)
        with self.assertRaises(Exception) as raised:
            client.search_offers(
                pricing_type="on_demand",
                gpu_names=["B200"],
            )
        self.assertNotIn(key, str(raised.exception))
        self.assertEqual(raised.exception.request_count, 1)

    def test_auth_error_never_contains_api_key(self):
        key = "must-not-leak"
        session = FakeSession([FakeResponse(status_code=401, body={})])
        client = VastAIClient(key, session=session)
        with self.assertRaises(VastAIAuthenticationError) as raised:
            client.search_offers(
                pricing_type="on_demand",
                gpu_names=["B200"],
            )
        self.assertNotIn(key, str(raised.exception))

    def test_schema_change_fails_clearly(self):
        client = VastAIClient(
            "secret",
            session=FakeSession([FakeResponse(body={"items": []})]),
        )
        with self.assertRaisesRegex(VastAISchemaError, "offers list"):
            client.search_offers(
                pricing_type="on_demand",
                gpu_names=["B200"],
            )

    def test_saturated_multi_name_query_is_partitioned_and_deduplicated(self):
        session = FakeSession(
            [
                FakeResponse(body={"offers": [offer(90), offer(91)]}),
                FakeResponse(body={"offers": [offer(1)]}),
                FakeResponse(body={"offers": [offer(1), offer(2, "L40S")]}),
            ]
        )
        client = VastAIClient("secret", session=session)
        result = client.search_offers(
            pricing_type="on_demand",
            gpu_names=["H100 SXM", "L40S"],
            limit=2,
        )
        self.assertEqual(result.request_count, 3)
        self.assertEqual({row["id"] for row in result.offers}, {1, 2})
        self.assertTrue(result.results_truncated)
        self.assertIn("visible inventory may be partial", result.warnings[0])
        queried_names = [call[1]["json"]["gpu_name"]["in"] for call in session.calls]
        self.assertEqual(
            queried_names,
            [["H100 SXM", "L40S"], ["H100 SXM"], ["L40S"]],
        )

    def test_normalization_preserves_price_meaning_and_unknowns(self):
        rows, warnings = normalize_vast_offers(
            [offer(1), offer(2, "Future GPU", dph_total=None, num_gpus=None)],
            pricing_type="interruptible",
            snapshot_timestamp=datetime(2026, 7, 31, 12, 34, 56, tzinfo=UTC),
        )
        first = rows.loc[rows["offer_id"].eq(1)].iloc[0]
        self.assertEqual(first["instance_price_per_hour_usd"], 4.0)
        self.assertEqual(first["price_per_gpu_hour_usd"], 2.0)
        self.assertEqual(first["interruptible_price_per_gpu_hour_usd"], 2.0)
        self.assertEqual(first["min_bid_price_per_hour_usd"], 2.0)
        self.assertEqual(first["country"], "US")
        unknown = rows.loc[rows["offer_id"].eq(2)].iloc[0]
        self.assertEqual(unknown["gpu_model"], "FUTURE_GPU")
        self.assertTrue(unknown[["gpu_count", "price_per_gpu_hour_usd"]].isna().all())
        self.assertTrue(any("schema_warning" in warning for warning in warnings))
        self.assertTrue(any("unrecognized" in warning for warning in warnings))

    def test_model_mapping_does_not_guess_h200_form_factor(self):
        self.assertEqual(canonicalize_gpu_model("H100 SXM")[0], "H100_SXM")
        self.assertEqual(canonicalize_gpu_model("A100 PCIE", 81920)[0], "A100_80GB")
        self.assertNotEqual(canonicalize_gpu_model("A100 PCIE", None)[0], "A100_80GB")
        model, warning = canonicalize_gpu_model("H200")
        self.assertEqual(model, "H200")
        self.assertIn("not classified as H200_SXM", warning)


if __name__ == "__main__":
    unittest.main()

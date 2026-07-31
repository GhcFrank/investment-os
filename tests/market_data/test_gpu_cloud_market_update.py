import os
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from market_data import update_gpu_cloud_market as update_module
from market_data.gpu_cloud_config import (
    GPU_CLOUD_FETCH_LOG_COLUMNS,
    GPU_CLOUD_HISTORY_COLUMNS,
)
from market_data.gpu_cloud_status import (
    NO_MARKET_DATA,
    PROVIDER_ERROR,
    SCHEMA_ERROR,
    SUCCESS_WITH_WARNINGS,
)
from market_data.update_gpu_cloud_market import run_gpu_cloud_market_update
from market_data.vast_ai_client import (
    VastAIAuthenticationError,
    VastAIResponseError,
    VastAISchemaError,
    VastSearchResult,
)


def source_offer(offer_id, pricing_type):
    return {
        "id": offer_id,
        "machine_id": offer_id + 100,
        "host_id": offer_id + 200,
        "geolocation": "Toronto, CA",
        "gpu_name": "H100 SXM",
        "gpu_ram": 81920,
        "num_gpus": 2,
        "dph_total": 4.0 if pricing_type == "on_demand" else 2.0,
        "min_bid": 1.8,
        "rentable": True,
        "rented": False,
        "verification": "verified",
        "reliability": 0.99,
    }


class FakeClient:
    def __init__(self, failures=None, secret_error="", warnings=(), empty=False):
        self.failures = set(failures or [])
        self.secret_error = secret_error
        self.warnings = tuple(warnings)
        self.empty = empty
        self.calls = []

    def search_offers(self, *, pricing_type, gpu_names):
        self.calls.append((pricing_type, tuple(gpu_names)))
        if pricing_type in self.failures:
            raise RuntimeError(self.secret_error or "provider failed")
        return VastSearchResult(
            offers=(
                ()
                if self.empty
                else (
                    source_offer(
                        1 if pricing_type == "on_demand" else 2,
                        pricing_type,
                    ),
                )
            ),
            warnings=self.warnings,
            request_count=1,
            results_truncated=False,
        )


class GPUCloudMarketUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.history = root / "history.csv"
        self.fetch_log = root / "fetch.csv"
        self.now = datetime(2026, 7, 31, 12, 34, 56, tzinfo=UTC)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_update_is_minute_keyed_atomic_and_idempotent(self):
        client = FakeClient()
        first = run_gpu_cloud_market_update(
            client=client,
            history_file=self.history,
            fetch_log_file=self.fetch_log,
            now=self.now,
        )
        first_history = self.history.read_bytes()
        first_log = self.fetch_log.read_bytes()
        second = run_gpu_cloud_market_update(
            client=client,
            history_file=self.history,
            fetch_log_file=self.fetch_log,
            now=self.now,
        )

        self.assertTrue(first.history_file_written)
        self.assertFalse(second.history_file_written)
        self.assertFalse(second.fetch_log_file_written)
        self.assertEqual(first_history, self.history.read_bytes())
        self.assertEqual(first_log, self.fetch_log.read_bytes())
        history = pd.read_csv(self.history)
        self.assertEqual(list(history.columns), GPU_CLOUD_HISTORY_COLUMNS)
        self.assertEqual(len(history), 2)
        self.assertEqual(set(history["pricing_type"]), {"on_demand", "interruptible"})
        self.assertEqual(
            set(history["snapshot_timestamp_utc"]), {"2026-07-31T12:34:00Z"}
        )
        self.assertFalse(history.duplicated(
            ["snapshot_timestamp_utc", "provider", "offer_id", "pricing_type"]
        ).any())

    def test_partial_failure_is_logged_without_fabricating_zero_inventory(self):
        summary = run_gpu_cloud_market_update(
            client=FakeClient(failures={"interruptible"}),
            history_file=self.history,
            fetch_log_file=self.fetch_log,
            now=self.now,
        )
        self.assertEqual(summary.pricing_types_succeeded, 1)
        self.assertEqual(summary.pricing_types_failed, 1)
        history = pd.read_csv(self.history)
        self.assertEqual(set(history["pricing_type"]), {"on_demand"})
        fetch_log = pd.read_csv(self.fetch_log)
        statuses = dict(zip(fetch_log["pricing_type"], fetch_log["status"]))
        self.assertEqual(statuses["on_demand"], "SUCCESS")
        self.assertEqual(statuses["interruptible"], PROVIDER_ERROR)

    def test_all_failures_leave_existing_history_untouched_and_return_nonzero(self):
        self.history.write_text(
            ",".join(GPU_CLOUD_HISTORY_COLUMNS) + "\n",
            encoding="utf-8",
        )
        before = self.history.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "No complete Vast.ai"):
            run_gpu_cloud_market_update(
                client=FakeClient(failures={"on_demand", "interruptible"}),
                history_file=self.history,
                fetch_log_file=self.fetch_log,
                now=self.now,
            )
        self.assertEqual(before, self.history.read_bytes())
        fetch_log = pd.read_csv(self.fetch_log)
        self.assertEqual(list(fetch_log.columns), GPU_CLOUD_FETCH_LOG_COLUMNS)
        self.assertEqual(set(fetch_log["status"]), {PROVIDER_ERROR})

    def test_missing_key_is_clear_and_makes_no_network_client(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                VastAIAuthenticationError, "VAST_API_KEY"
            ):
                run_gpu_cloud_market_update(
                    history_file=self.history,
                    fetch_log_file=self.fetch_log,
                    now=self.now,
                    load_environment=False,
                )
        fetch_log = pd.read_csv(self.fetch_log)
        self.assertEqual(set(fetch_log["status"]), {"API_KEY_MISSING"})
        self.assertFalse(self.history.exists())

    def test_in_process_pipeline_path_loads_project_dotenv(self):
        fake_client = FakeClient()

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(
                update_module,
                "get_project_environment_value",
                return_value="loaded-for-test",
            ) as environment_value,
            patch.object(
                update_module,
                "VastAIClient",
                return_value=fake_client,
            ) as client_class,
        ):
            summary = update_module.run_gpu_cloud_market_update(
                history_file=self.history,
                fetch_log_file=self.fetch_log,
                now=self.now,
            )
        environment_value.assert_called_once_with(
            "VAST_API_KEY",
            env_file=None,
            load_environment=True,
        )
        client_class.assert_called_once_with("loaded-for-test")
        self.assertEqual(summary.pricing_types_succeeded, 2)

    def test_warnings_always_produce_success_with_warnings(self):
        warning = "gpu_model_warning: H200 form factor is not specified"
        run_gpu_cloud_market_update(
            client=FakeClient(warnings=(warning, warning)),
            history_file=self.history,
            fetch_log_file=self.fetch_log,
            now=self.now,
        )
        fetch_log = pd.read_csv(self.fetch_log)
        self.assertEqual(
            set(fetch_log["status"]),
            {SUCCESS_WITH_WARNINGS},
        )
        self.assertTrue(fetch_log["data_quality_notes"].eq(warning).all())

    def test_valid_empty_response_is_no_market_data(self):
        summary = run_gpu_cloud_market_update(
            client=FakeClient(empty=True),
            history_file=self.history,
            fetch_log_file=self.fetch_log,
            now=self.now,
        )
        fetch_log = pd.read_csv(self.fetch_log)
        self.assertEqual(set(fetch_log["status"]), {NO_MARKET_DATA})
        self.assertEqual(summary.pricing_types_succeeded, 2)
        self.assertFalse(self.history.exists())

    def test_schema_and_provider_errors_have_distinct_status_and_attempts(self):
        class ErrorClient:
            def search_offers(self, *, pricing_type, gpu_names):
                if pricing_type == "on_demand":
                    raise VastAISchemaError("bad schema", request_count=1)
                raise VastAIResponseError("timeout", request_count=3)

        with self.assertRaisesRegex(RuntimeError, "No complete Vast.ai"):
            run_gpu_cloud_market_update(
                client=ErrorClient(),
                history_file=self.history,
                fetch_log_file=self.fetch_log,
                now=self.now,
            )
        fetch_log = pd.read_csv(self.fetch_log).set_index("pricing_type")
        self.assertEqual(fetch_log.loc["on_demand", "status"], SCHEMA_ERROR)
        self.assertEqual(
            fetch_log.loc["interruptible", "status"], PROVIDER_ERROR
        )
        self.assertEqual(fetch_log.loc["on_demand", "request_count"], 1)
        self.assertEqual(fetch_log.loc["interruptible", "request_count"], 3)

    def test_arbitrary_provider_exception_cannot_write_secret_to_files(self):
        secret = "super-secret-token"
        with self.assertRaises(RuntimeError):
            run_gpu_cloud_market_update(
                client=FakeClient(
                    failures={"on_demand", "interruptible"},
                    secret_error=secret,
                ),
                history_file=self.history,
                fetch_log_file=self.fetch_log,
                now=self.now,
            )
        self.assertNotIn(secret, self.fetch_log.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

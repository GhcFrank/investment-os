import unittest

from market_data.gpu_cloud_status import (
    API_KEY_MISSING,
    NO_MARKET_DATA,
    PROVIDER_ERROR,
    SCHEMA_ERROR,
    SUCCESS,
    SUCCESS_WITH_WARNINGS,
    determine_snapshot_status,
    is_snapshot_eligible_for_signals,
)


class GPUCloudSnapshotStatusTests(unittest.TestCase):
    def decision(self, **overrides):
        values = {
            "api_key_configured": True,
            "request_succeeded": True,
            "schema_valid": True,
            "valid_offer_count": 2,
            "source_offer_count": 2,
            "warnings": [],
            "results_truncated": False,
        }
        values.update(overrides)
        return determine_snapshot_status(**values)

    def test_status_priority_and_complete_outcomes(self):
        self.assertEqual(
            self.decision(api_key_configured=False).status,
            API_KEY_MISSING,
        )
        self.assertEqual(
            self.decision(request_succeeded=False).status,
            PROVIDER_ERROR,
        )
        self.assertEqual(
            self.decision(schema_valid=False).status,
            SCHEMA_ERROR,
        )
        self.assertEqual(
            self.decision(
                valid_offer_count=0,
                source_offer_count=0,
            ).status,
            NO_MARKET_DATA,
        )
        self.assertEqual(self.decision().status, SUCCESS)
        self.assertEqual(
            self.decision(
                warnings=[
                    "gpu_model_warning: Vast.ai H200 form factor is not specified"
                ]
            ).status,
            SUCCESS_WITH_WARNINGS,
        )

    def test_all_unusable_source_offers_are_schema_error(self):
        decision = self.decision(
            valid_offer_count=0,
            source_offer_count=3,
            warnings=["schema_warning: invalid offer"],
        )
        self.assertEqual(decision.status, SCHEMA_ERROR)

    def test_truncated_result_is_not_eligible_success(self):
        decision = self.decision(results_truncated=True)
        self.assertEqual(decision.status, PROVIDER_ERROR)
        self.assertIn("truncated", decision.data_quality_notes)

    def test_warning_deduplication_and_order_are_stable(self):
        warnings_a = [" warning B ", "warning A", "warning B"]
        warnings_b = ["warning B", "warning A", "warning A"]
        decision_a = self.decision(warnings=warnings_a)
        decision_b = self.decision(warnings=warnings_b)
        self.assertEqual(decision_a.status, SUCCESS_WITH_WARNINGS)
        self.assertEqual(decision_a, decision_b)
        self.assertEqual(decision_a.warnings, ("warning A", "warning B"))
        self.assertEqual(
            decision_a.data_quality_notes,
            "warning A; warning B",
        )

    def test_signal_eligibility_including_legacy_partial_and_real_zero(self):
        base = {
            "request_count": 1,
            "offer_count": 2,
            "results_truncated": False,
        }
        for status in (SUCCESS, SUCCESS_WITH_WARNINGS):
            self.assertTrue(
                is_snapshot_eligible_for_signals({**base, "status": status})
            )
        for status in (API_KEY_MISSING, PROVIDER_ERROR, SCHEMA_ERROR):
            self.assertFalse(
                is_snapshot_eligible_for_signals({**base, "status": status})
            )
        self.assertTrue(
            is_snapshot_eligible_for_signals({**base, "status": "PARTIAL"})
        )
        self.assertFalse(
            is_snapshot_eligible_for_signals(
                {**base, "status": "PARTIAL", "request_count": 0}
            )
        )
        self.assertFalse(
            is_snapshot_eligible_for_signals(
                {**base, "status": "PARTIAL", "results_truncated": True}
            )
        )
        self.assertTrue(
            is_snapshot_eligible_for_signals(
                {
                    **base,
                    "status": NO_MARKET_DATA,
                    "offer_count": 0,
                }
            )
        )


if __name__ == "__main__":
    unittest.main()

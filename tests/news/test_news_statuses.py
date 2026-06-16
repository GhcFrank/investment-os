import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from news import fetch_semiengineering_news as fetch
from news.news_filter import (
    MANUAL_DECISION_COLUMNS,
    NEWS_COLUMNS,
    REJECT_COLUMNS,
    REVIEW_COLUMNS,
    assert_no_cross_status_overlap,
    reconcile_news_statuses,
)
from news.news_utils import atomic_write_csv, read_csv_safe


def news_row(news_id: str, status: str, first_seen: str, last_seen: str) -> dict[str, str]:
    row = {column: "" for column in NEWS_COLUMNS}
    row.update(
        {
            "news_id": news_id,
            "source_id": "semiengineering",
            "source_post_id": news_id.rsplit("_", 1)[-1],
            "published_at_gmt": "2026-06-01T00:00:00Z",
            "title": f"Title {news_id}",
            "url": f"https://example.test/{news_id}",
            "filter_status": status,
            "rule_filter_status": "review" if status.startswith("manual_") else status,
            "filter_reason": status,
            "first_seen_at": first_seen,
            "last_seen_at": last_seen,
            "filter_rule_version": "test",
        }
    )
    return row


class NewsStatusTests(unittest.TestCase):
    def test_reconcile_transitions_and_preserves_first_seen(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            history = tmp_path / "history.csv"
            review = tmp_path / "review.csv"
            reject = tmp_path / "reject.csv"

            atomic_write_csv(
                pd.DataFrame([news_row("semiengineering_1", "keep", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")]),
                history,
                NEWS_COLUMNS,
            )
            old_review = {**news_row("semiengineering_2", "review", "2026-01-02T00:00:00Z", "2026-01-02T00:00:00Z"), "manual_notes": "watch"}
            atomic_write_csv(pd.DataFrame([old_review]), review, REVIEW_COLUMNS)
            atomic_write_csv(
                pd.DataFrame([news_row("semiengineering_3", "reject", "2026-01-03T00:00:00Z", "2026-01-03T00:00:00Z")]).reindex(columns=REJECT_COLUMNS),
                reject,
                REJECT_COLUMNS,
            )

            processed = pd.DataFrame(
                [
                    news_row("semiengineering_2", "keep", "2026-06-01T00:00:00Z", "2026-06-01T00:00:00Z"),
                    news_row("semiengineering_3", "review", "2026-06-01T00:00:00Z", "2026-06-01T00:00:00Z"),
                ],
                columns=NEWS_COLUMNS,
            )

            keep_df, review_df, reject_df = reconcile_news_statuses(
                processed,
                history,
                review,
                reject,
            )

            self.assertIn("semiengineering_2", set(keep_df["news_id"]))
            self.assertNotIn("semiengineering_2", set(review_df["news_id"]))
            self.assertIn("semiengineering_3", set(review_df["news_id"]))
            self.assertNotIn("semiengineering_3", set(reject_df["news_id"]))
            first_seen = keep_df.set_index("news_id").loc["semiengineering_2", "first_seen_at"]
            self.assertEqual(first_seen, "2026-01-02T00:00:00Z")

    def test_reconcile_preserves_review_manual_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            history = tmp_path / "history.csv"
            review = tmp_path / "review.csv"
            reject = tmp_path / "reject.csv"
            old_review = {
                **news_row("semiengineering_4", "review", "2026-01-04T00:00:00Z", "2026-01-04T00:00:00Z"),
                "manual_decision": "",
                "manual_notes": "needs domain check",
                "reviewed_at": "2026-01-05T00:00:00Z",
            }
            atomic_write_csv(pd.DataFrame([old_review]), review, REVIEW_COLUMNS)

            processed = pd.DataFrame(
                [news_row("semiengineering_4", "review", "2026-06-01T00:00:00Z", "2026-06-01T00:00:00Z")],
                columns=NEWS_COLUMNS,
            )
            _, review_df, _ = reconcile_news_statuses(processed, history, review, reject)
            row = review_df.set_index("news_id").loc["semiengineering_4"]

            self.assertEqual(row["manual_notes"], "needs domain check")
            self.assertEqual(row["reviewed_at"], "2026-01-05T00:00:00Z")

    def test_reconcile_handles_missing_and_empty_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            reject = tmp_path / "reject.csv"
            reject.write_text("", encoding="utf-8")
            processed = pd.DataFrame(
                [news_row("semiengineering_5", "manual_keep", "2026-06-01T00:00:00Z", "2026-06-01T00:00:00Z")],
                columns=NEWS_COLUMNS,
            )

            keep_df, review_df, reject_df = reconcile_news_statuses(
                processed,
                tmp_path / "history.csv",
                tmp_path / "review.csv",
                reject,
            )

            self.assertEqual(set(keep_df["news_id"]), {"semiengineering_5"})
            self.assertTrue(review_df.empty)
            self.assertTrue(reject_df.empty)

    def test_cross_status_overlap_assertion(self):
        keep = pd.DataFrame([{"news_id": "same"}])
        review = pd.DataFrame([{"news_id": "same"}])
        reject = pd.DataFrame(columns=["news_id"])

        with self.assertRaises(RuntimeError):
            assert_no_cross_status_overlap(keep, review, reject)


class ManualDecisionTests(unittest.TestCase):
    def test_apply_review_decisions_persists_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = {
                "CURRENT_NEWS_FILE": tmp_path / "current.csv",
                "HISTORY_FILE": tmp_path / "history.csv",
                "REVIEW_FILE": tmp_path / "review.csv",
                "REJECT_FILE": tmp_path / "reject.csv",
                "FETCH_LOG_FILE": tmp_path / "fetch_log.csv",
                "MANUAL_DECISIONS_FILE": tmp_path / "manual.csv",
            }
            review_row = {
                **news_row("semiengineering_6", "review", "2026-06-01T00:00:00Z", "2026-06-01T00:00:00Z"),
                "manual_decision": "keep",
                "manual_notes": "important capacity signal",
                "reviewed_at": "2026-06-02T00:00:00Z",
            }
            atomic_write_csv(pd.DataFrame([review_row]), paths["REVIEW_FILE"], REVIEW_COLUMNS)

            with patch.object(fetch, "CURRENT_NEWS_FILE", paths["CURRENT_NEWS_FILE"]):
                with patch.object(fetch, "HISTORY_FILE", paths["HISTORY_FILE"]):
                    with patch.object(fetch, "REVIEW_FILE", paths["REVIEW_FILE"]):
                        with patch.object(fetch, "REJECT_FILE", paths["REJECT_FILE"]):
                            with patch.object(fetch, "FETCH_LOG_FILE", paths["FETCH_LOG_FILE"]):
                                with patch.object(fetch, "MANUAL_DECISIONS_FILE", paths["MANUAL_DECISIONS_FILE"]):
                                    first_exit = fetch._apply_review_decisions()
                                    second_exit = fetch._apply_review_decisions()

            self.assertEqual(first_exit, 0)
            self.assertEqual(second_exit, 0)

            manual = read_csv_safe(paths["MANUAL_DECISIONS_FILE"], MANUAL_DECISION_COLUMNS)
            history = read_csv_safe(paths["HISTORY_FILE"], NEWS_COLUMNS)
            review = read_csv_safe(paths["REVIEW_FILE"], REVIEW_COLUMNS)

            self.assertEqual(len(manual), 1)
            self.assertEqual(manual.loc[0, "manual_notes"], "important capacity signal")
            self.assertEqual(history.loc[0, "filter_status"], "manual_keep")
            self.assertEqual(history.loc[0, "manual_override"], "True")
            self.assertTrue(review.empty)

import unittest
from pathlib import Path

import pandas as pd

from news import deepseek_news_followup as followup


def flash_row(news_id: str, decision: str, status: str = "ok") -> dict[str, str]:
    return {
        "news_id": news_id,
        "model": "deepseek-v4-flash",
        "prompt_version": "deepseek_flash_preprocess_v1",
        "status": status,
        "recommended_decision": decision,
        "summary": f"{news_id} summary",
        "why_it_matters": f"{news_id} why",
    }


class DeepSeekFollowupSelectionTests(unittest.TestCase):
    def test_selects_only_flash_keep_watch_ok_rows(self):
        analysis = pd.DataFrame(
            [
                flash_row("A", "keep"),
                flash_row("B", "watch"),
                flash_row("C", "reject"),
                flash_row("D", "keep", status="failed"),
            ]
        )
        review_queue = pd.DataFrame(
            [
                {"news_id": "A", "title": "A title"},
                {"news_id": "B", "title": "B title"},
                {"news_id": "C", "title": "C title"},
                {"news_id": "D", "title": "D title"},
            ]
        )

        selected = followup.load_flash_followup_candidates(
            analysis=analysis,
            review_queue=review_queue,
            decision_filter=["keep", "watch"],
        )

        self.assertEqual(list(selected["news_id"]), ["A", "B"])

    def test_default_skip_logic_skips_ok_and_failed(self):
        candidates = pd.DataFrame([flash_row("A", "keep"), flash_row("B", "watch")])
        existing = pd.DataFrame(
            [
                {
                    "news_id": "A",
                    "followup_model": "deepseek-v4-pro",
                    "followup_prompt_version": "deepseek_pro_followup_v1",
                    "status": "ok",
                },
                {
                    "news_id": "B",
                    "followup_model": "deepseek-v4-pro",
                    "followup_prompt_version": "deepseek_pro_followup_v1",
                    "status": "failed",
                },
            ]
        )

        selected, skipped = followup.select_followup_candidates(
            candidates=candidates,
            existing=existing,
            followup_model="deepseek-v4-pro",
            followup_prompt_version="deepseek_pro_followup_v1",
            limit=10,
        )

        self.assertTrue(selected.empty)
        self.assertEqual(skipped, 2)

    def test_retry_failed_retries_only_failed(self):
        candidates = pd.DataFrame([flash_row("A", "keep"), flash_row("B", "watch")])
        existing = pd.DataFrame(
            [
                {
                    "news_id": "A",
                    "followup_model": "deepseek-v4-pro",
                    "followup_prompt_version": "deepseek_pro_followup_v1",
                    "status": "ok",
                },
                {
                    "news_id": "B",
                    "followup_model": "deepseek-v4-pro",
                    "followup_prompt_version": "deepseek_pro_followup_v1",
                    "status": "failed",
                },
            ]
        )

        selected, skipped = followup.select_followup_candidates(
            candidates=candidates,
            existing=existing,
            followup_model="deepseek-v4-pro",
            followup_prompt_version="deepseek_pro_followup_v1",
            limit=10,
            retry_failed=True,
        )

        self.assertEqual(list(selected["news_id"]), ["B"])
        self.assertEqual(skipped, 1)


class DeepSeekWorkflowTests(unittest.TestCase):
    def test_weekly_workflow_uses_flash_manifest_and_skip_existing_attempts(self):
        workflow = Path(".github/workflows/weekly_news.yml").read_text(encoding="utf-8")

        self.assertIn("deepseek-v4-flash", workflow)
        self.assertIn("--news-ids-file data/news/news_latest_fetch_manifest.csv", workflow)
        self.assertIn("--skip-existing-attempts", workflow)
        self.assertNotIn("deepseek-v4-pro", workflow)
        self.assertNotIn("--force", workflow)

    def test_manual_followup_workflow_is_dispatch_only_and_uses_pro(self):
        workflow = Path(".github/workflows/manual_deepseek_news_followup.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("workflow_dispatch", workflow)
        self.assertNotIn("schedule:", workflow)
        self.assertIn("deepseek-v4-pro", workflow)
        self.assertIn("review_keep_watch_with_deepseek.py", workflow)


if __name__ == "__main__":
    unittest.main()


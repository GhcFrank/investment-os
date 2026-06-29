import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from news import fetch_semiengineering_news as fetch
from news.news_utils import repo_relative_path


class CliModeTests(unittest.TestCase):
    def test_apply_review_and_dry_run_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit) as error:
            fetch.parse_args(["--apply-review-decisions", "--dry-run"])

        self.assertEqual(error.exception.code, 2)

    def test_apply_review_and_reprocess_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit) as error:
            fetch.parse_args(
                [
                    "--apply-review-decisions",
                    "--reprocess-raw",
                    "data/news/raw/semiengineering/sample.json",
                ]
            )

        self.assertEqual(error.exception.code, 2)

    def test_reprocess_and_dry_run_are_allowed(self):
        args = fetch.parse_args(
            [
                "--reprocess-raw",
                "data/news/raw/semiengineering/sample.json",
                "--dry-run",
            ]
        )

        self.assertTrue(args.dry_run)
        self.assertEqual(args.reprocess_raw, "data/news/raw/semiengineering/sample.json")

    def test_api_fetch_and_dry_run_are_allowed(self):
        args = fetch.parse_args(["--dry-run"])

        self.assertTrue(args.dry_run)
        self.assertIsNone(args.reprocess_raw)
        self.assertFalse(args.apply_review_decisions)


class FetchTaxonomyTests(unittest.TestCase):
    def test_collect_post_term_ids(self):
        category_ids, tag_ids = fetch.collect_post_term_ids(
            [
                {"categories": [1, "2", "bad"], "tags": [10, "11"]},
                {"categories": [2, 3], "tags": [11, None]},
            ]
        )

        self.assertEqual(category_ids, {1, 2, 3})
        self.assertEqual(tag_ids, {10, 11})

    def test_fetch_terms_by_ids_only_requests_used_ids(self):
        calls = []

        def fake_fetch_paginated(endpoint, params, label):
            calls.append((endpoint, params, label))
            return ([{"id": term_id, "name": f"Term {term_id}", "slug": str(term_id)} for term_id in params["include"].split(",")], 1, 1, 1)

        with patch.object(fetch, "fetch_paginated", side_effect=fake_fetch_paginated):
            terms, pages = fetch.fetch_terms_by_ids("https://example.test/tags", {5, 2, 9}, batch_size=2)

        self.assertEqual(pages, 2)
        self.assertEqual([call[1]["include"] for call in calls], ["2,5", "9"])
        self.assertEqual({term["id"] for term in terms}, {"2", "5", "9"})

    def test_term_cache_merges_and_decodes_html(self):
        existing = pd.DataFrame(
            [{"term_id": "1", "name": "Old", "slug": "old", "count": "1", "updated_at": "old"}],
            columns=fetch.TERM_COLUMNS,
        )
        new_terms = fetch.term_rows(
            [
                {"id": 1, "name": "R&amp;D", "slug": "r-d", "count": 2},
                {"id": 2, "name": "AI&nbsp;Cloud", "slug": "ai-cloud", "count": 3},
            ],
            "new",
        )

        merged = fetch.upsert_term_cache(existing, new_terms)
        term_map = fetch.build_term_map(merged)

        self.assertEqual(term_map[1], "R&D")
        self.assertEqual(term_map[2], "AI Cloud")
        self.assertEqual(len(merged), 2)

    def test_repo_relative_raw_path(self):
        raw_path = fetch.BASE_DIR / "data" / "news" / "raw" / "semiengineering" / "sample.json"

        self.assertEqual(
            repo_relative_path(raw_path, fetch.BASE_DIR),
            "data/news/raw/semiengineering/sample.json",
        )


class LatestFetchManifestTests(unittest.TestCase):
    def test_manifest_marks_new_news_id(self):
        rows = pd.DataFrame(
            [
                {
                    "news_id": "A",
                    "source_id": "semiengineering",
                    "title": "New Article",
                    "url": "https://example.test/a",
                    "published_at_gmt": "2026-06-29",
                    "modified_at_gmt": "2026-06-29",
                }
            ]
        )

        manifest = fetch.build_latest_fetch_manifest(rows, [], "2026-06-29T00:00:00Z")

        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest.loc[0, "news_id"], "A")
        self.assertEqual(manifest.loc[0, "change_status"], "new")

    def test_manifest_omits_unchanged_existing_news_id(self):
        row = {
            "news_id": "A",
            "source_id": "semiengineering",
            "title": "Same Article",
            "excerpt": "same",
            "url": "https://example.test/a",
            "published_at_gmt": "2026-06-29",
            "modified_at_gmt": "2026-06-29",
        }
        rows = pd.DataFrame([row])

        manifest = fetch.build_latest_fetch_manifest(
            rows,
            [pd.DataFrame([row])],
            "2026-06-29T00:00:00Z",
        )

        self.assertTrue(manifest.empty)

    def test_manifest_marks_updated_news_id(self):
        old = {
            "news_id": "A",
            "source_id": "semiengineering",
            "title": "Old Article",
            "excerpt": "old",
            "url": "https://example.test/a",
            "published_at_gmt": "2026-06-29",
            "modified_at_gmt": "2026-06-29",
        }
        new = {**old, "title": "Updated Article"}

        manifest = fetch.build_latest_fetch_manifest(
            pd.DataFrame([new]),
            [pd.DataFrame([old])],
            "2026-06-29T00:00:00Z",
        )

        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest.loc[0, "change_status"], "updated")


class DryRunTests(unittest.TestCase):
    def _patch_output_paths(self, tmp_path: Path):
        return [
            patch.object(fetch, "CURRENT_NEWS_FILE", tmp_path / "semiengineering_news.csv"),
            patch.object(fetch, "HISTORY_FILE", tmp_path / "semiengineering_news_history.csv"),
            patch.object(fetch, "REVIEW_FILE", tmp_path / "news_review_queue.csv"),
            patch.object(fetch, "REJECT_FILE", tmp_path / "news_rejected_log.csv"),
            patch.object(fetch, "FETCH_LOG_FILE", tmp_path / "news_fetch_log.csv"),
            patch.object(fetch, "MANUAL_DECISIONS_FILE", tmp_path / "news_manual_decisions.csv"),
            patch.object(fetch, "CATEGORIES_FILE", tmp_path / "reference" / "categories.csv"),
            patch.object(fetch, "TAGS_FILE", tmp_path / "reference" / "tags.csv"),
            patch.object(fetch, "RAW_DIR", tmp_path / "raw" / "semiengineering"),
        ]

    def test_dry_run_writes_no_files(self):
        post = {
            "id": 501,
            "date": "2026-06-01T00:00:00",
            "date_gmt": "2026-06-01T04:00:00",
            "modified_gmt": "2026-06-01T04:05:00",
            "link": "https://semiengineering.com/test/",
            "title": {"rendered": "1 Megawatt Racks In Data Centers"},
            "excerpt": {"rendered": "Power delivery for rack scale AI."},
            "author": 1,
            "categories": [11],
            "tags": [101],
        }
        args = argparse.Namespace(
            apply_review_decisions=False,
            dry_run=True,
            reprocess_raw=None,
            lookback_days=30,
            as_of="2026-06-15T00:00:00Z",
            save_raw=True,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            contexts = self._patch_output_paths(tmp_path)

            with contexts[0], contexts[1], contexts[2], contexts[3], contexts[4], contexts[5], contexts[6], contexts[7], contexts[8]:
                with patch.object(fetch, "fetch_posts", return_value=([post], 1, 1, 1)):
                    with patch.object(fetch, "fetch_terms_by_ids", side_effect=[([{"id": 11, "name": "Top Stories", "slug": "top-stories", "count": 1}], 1), ([{"id": 101, "name": "Power", "slug": "power", "count": 1}], 1)]):
                        exit_code = fetch.run_backfill(args)

            self.assertEqual(exit_code, 0)
            self.assertEqual(list(tmp_path.rglob("*")), [])

    def test_failed_dry_run_does_not_write_fetch_log(self):
        args = argparse.Namespace(
            apply_review_decisions=False,
            dry_run=True,
            reprocess_raw=None,
            lookback_days=30,
            as_of="2026-06-15T00:00:00Z",
            save_raw=True,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            contexts = self._patch_output_paths(tmp_path)

            with contexts[0], contexts[1], contexts[2], contexts[3], contexts[4], contexts[5], contexts[6], contexts[7], contexts[8]:
                with patch.object(fetch, "fetch_posts", side_effect=RuntimeError("boom")):
                    exit_code = fetch.run_backfill(args)

            self.assertEqual(exit_code, 1)
            self.assertFalse((tmp_path / "news_fetch_log.csv").exists())

    def test_raw_payload_fixture_has_no_content_rendered(self):
        fixture = Path(__file__).parent / "fixtures" / "semiengineering_sample.json"
        payload = json.loads(fixture.read_text(encoding="utf-8"))

        self.assertGreaterEqual(len(payload["posts"]), 5)
        self.assertFalse(any("content" in post for post in payload["posts"]))

import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from news import fetch_semiengineering_news as fetch
from news.news_filter import (
    FILTER_RULE_VERSION,
    MANUAL_DECISION_COLUMNS,
    NEWS_COLUMNS,
    REJECT_COLUMNS,
    REVIEW_COLUMNS,
    assert_no_cross_status_overlap,
    load_filter_config,
    rows_from_posts,
)
from news.news_utils import atomic_write_csv


def post(post_id: int, title: str, excerpt: str, categories=None, tags=None) -> dict:
    return {
        "id": post_id,
        "date": "2026-06-01T00:00:00",
        "date_gmt": "2026-06-01T07:00:00",
        "modified_gmt": "2026-06-01T07:10:00",
        "link": f"https://semiengineering.com/{post_id}/",
        "title": {"rendered": title},
        "excerpt": {"rendered": excerpt},
        "author": 1,
        "categories": categories or [1],
        "tags": tags or [],
    }


class NewsReprocessRegressionTests(unittest.TestCase):
    def test_reprocess_fixture_covers_matching_roundup_version_and_manual_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manual_file = tmp_path / "manual.csv"
            paths = {
                "CURRENT_NEWS_FILE": tmp_path / "current.csv",
                "HISTORY_FILE": tmp_path / "history.csv",
                "REVIEW_FILE": tmp_path / "review.csv",
                "REJECT_FILE": tmp_path / "reject.csv",
            }
            manual = pd.DataFrame(
                [
                    {
                        "news_id": "semiengineering_2006",
                        "manual_decision": "keep",
                        "manual_notes": "manual promote",
                        "reviewed_at": "2026-06-02T00:00:00Z",
                        "applied_at": "2026-06-02T01:00:00Z",
                    },
                    {
                        "news_id": "semiengineering_2007",
                        "manual_decision": "reject",
                        "manual_notes": "manual reject",
                        "reviewed_at": "2026-06-02T00:00:00Z",
                        "applied_at": "2026-06-02T01:00:00Z",
                    },
                ],
                columns=MANUAL_DECISION_COLUMNS,
            )
            atomic_write_csv(manual, manual_file, MANUAL_DECISION_COLUMNS)
            config = load_filter_config(manual_decisions_file=manual_file)
            posts = [
                post(2001, "Micron Technology Expands HBM", "New DRAM capacity for AI infrastructure."),
                post(2002, "Boosting EUV Conversion Efficiency With 2-Micron Laser", "Lithography source power improves."),
                post(2003, "Blog Review: May 20", "AI and semiconductor links from the week."),
                post(2004, "Co-Packaged Optics Testing Faces Steep Data Center Ramp", "CPO and optical interconnect test demand rises."),
                post(2005, "New Reliability Bottleneck For AI Accelerator Packaging", "Qualification remains difficult."),
                post(2006, "Routine Supplier Newsletter", "Low relevance item promoted by manual review."),
                post(2007, "Micron Technology Expands HBM Capacity", "Manual rejection should override direct company match."),
            ]
            rows = rows_from_posts(
                posts=posts,
                category_map={1: "Top Stories"},
                tag_map={},
                seen_at="2026-06-02T00:00:00Z",
                config=config,
            )

            with ExitStack() as stack:
                for name, path in paths.items():
                    stack.enter_context(patch.object(fetch, name, path))

                keep, review, reject = fetch.write_outputs(rows)

            by_id = rows.set_index("news_id")
            self.assertIn("MU", by_id.loc["semiengineering_2001", "matched_tickers"])
            self.assertNotIn("MU", by_id.loc["semiengineering_2002", "matched_tickers"])
            self.assertEqual(by_id.loc["semiengineering_2003", "content_class"], "roundup")
            self.assertEqual(by_id.loc["semiengineering_2003", "source_quality"], "B")
            self.assertEqual(by_id.loc["semiengineering_2006", "filter_status"], "manual_keep")
            self.assertEqual(by_id.loc["semiengineering_2007", "filter_status"], "manual_reject")
            self.assertEqual(set(rows["filter_rule_version"]), {FILTER_RULE_VERSION})
            assert_no_cross_status_overlap(keep, review, reject)
            current = pd.read_csv(paths["CURRENT_NEWS_FILE"], dtype=str)
            self.assertEqual(
                set(current["news_id"]),
                set(keep["news_id"]),
            )


if __name__ == "__main__":
    unittest.main()

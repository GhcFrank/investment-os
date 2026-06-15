import unittest

from news.news_filter import filter_article, load_filter_config


class NewsFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_filter_config()

    def classify(self, title, excerpt="", category_names="", tag_names=""):
        return filter_article(
            {
                "title": title,
                "excerpt": excerpt,
                "category_names": category_names,
                "tag_names": tag_names,
            },
            config=self.config,
        )

    def test_direct_company_match(self):
        result = self.classify("Keysight Launches New 1.6T Optical Test Platform")

        self.assertIn("KEYS", result["matched_tickers"])
        self.assertEqual(result["filter_status"], "keep")

    def test_theme_match_without_company(self):
        result = self.classify("Challenges In Wafer-Level Silicon Photonics Packaging")

        self.assertTrue(
            "Optical" in result["matched_subthemes"]
            or "Optical Test & Metrology" in result["matched_subthemes"]
        )
        self.assertEqual(result["filter_status"], "keep")

    def test_generic_terms_do_not_keep(self):
        result = self.classify(
            "New Methods For Power Integrity",
            "chip design and testing",
        )

        self.assertNotEqual(result["filter_status"], "keep")

    def test_emerging_candidate(self):
        result = self.classify(
            "New Reliability Bottleneck For AI Accelerator Packaging"
        )

        self.assertEqual(result["emerging_candidate"], "True")
        self.assertEqual(result["filter_status"], "review")

    def test_common_tickers_do_not_false_match(self):
        result = self.classify("Teams Form New Industry Group")

        self.assertNotIn("TEAM", result["matched_tickers"])
        self.assertNotIn("FORM", result["matched_tickers"])

    def test_company_in_excerpt(self):
        result = self.classify(
            "New High-Speed Interconnect Platform",
            "The platform was introduced by Marvell.",
        )

        self.assertIn("MRVL", result["matched_tickers"])
        self.assertEqual(result["filter_status"], "keep")

    def test_promotional_exclusion(self):
        result = self.classify("Register Now For Sponsored Webinar On Verification")

        self.assertLess(result["exclusion_score"], 0)
        self.assertNotEqual(result["filter_status"], "keep")

    def test_case_and_hyphen_normalization(self):
        result = self.classify("CO-PACKAGED OPTICS FOR AI DATA CENTERS")

        self.assertIn("co packaged optics", result["matched_keywords"])
        self.assertEqual(result["filter_status"], "keep")


if __name__ == "__main__":
    unittest.main()

import unittest

from news.news_filter import classify_content, filter_article, load_filter_config


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

    def test_micron_requires_company_context(self):
        false_match = self.classify(
            "Improving Etch Control For 5 Micron Features",
            "Metrology teams compare linewidth variation across process windows.",
        )
        true_match = self.classify(
            "Micron Technology Advances HBM4 Memory",
            "The company discussed DRAM and high bandwidth memory roadmaps.",
        )

        self.assertNotIn("MU", false_match["matched_tickers"])
        self.assertIn("MU", true_match["matched_tickers"])

    def test_micron_technology_matches_mu(self):
        result = self.classify("Micron Technology Expands HBM4 Production")

        self.assertIn("MU", result["matched_tickers"])
        self.assertEqual(result["filter_status"], "keep")

    def test_micron_hbm_nearby_matches_mu(self):
        result = self.classify("Micron Plans New DRAM Capacity")

        self.assertIn("MU", result["matched_tickers"])
        self.assertEqual(result["filter_status"], "keep")

    def test_two_micron_laser_does_not_match_mu(self):
        result = self.classify(
            "Boosting EUV Conversion Efficiency With 2-Micron Dual-Beam Laser Irradiation"
        )

        self.assertNotIn("MU", result["matched_tickers"])

    def test_decimal_micron_lithography_does_not_match_mu(self):
        result = self.classify("Nikon Demonstrates 1.5 Micron L/S Lithography")

        self.assertNotIn("MU", result["matched_tickers"])

    def test_micron_scale_does_not_match_mu(self):
        result = self.classify("Micron-Scale Structures Improve Yield")

        self.assertNotIn("MU", result["matched_tickers"])

    def test_micron_ls_does_not_match_mu(self):
        result = self.classify("New Overlay Method For Micron L/S Process Control")

        self.assertNotIn("MU", result["matched_tickers"])

    def test_distant_hbm_does_not_turn_unit_micron_into_company(self):
        result = self.classify(
            "Lithography And Memory Market Update",
            "Nikon demonstrated 1.5 micron lithography. HBM prices rose elsewhere in the market.",
        )

        self.assertNotIn("MU", result["matched_tickers"])

    def test_coherent_requires_optical_context(self):
        false_match = self.classify(
            "A Coherent Approach To Verification",
            "EDA teams are improving coverage closure.",
        )
        true_match = self.classify(
            "Coherent Optical Transceiver Advances For AI Networks",
            "Photonics and optical interconnect demand continues to grow.",
        )

        self.assertNotIn("COHR", false_match["matched_tickers"])
        self.assertIn("COHR", true_match["matched_tickers"])

    def test_oracle_requires_cloud_context(self):
        false_match = self.classify(
            "EDA Teams Tune Oracle Rules",
            "A verification methodology article uses oracle checks generically.",
        )
        true_match = self.classify(
            "Oracle Cloud Infrastructure Expands AI Capacity",
            "OCI deployments add data center infrastructure for accelerators.",
        )

        self.assertNotIn("ORCL", false_match["matched_tickers"])
        self.assertIn("ORCL", true_match["matched_tickers"])

    def test_duplicate_aliases_score_once_per_ticker_field(self):
        result = self.classify("NVIDIA Blackwell Platform For AI Accelerators")

        self.assertIn("NVDA", result["matched_tickers"])
        self.assertEqual(result["company_score"], 8)

    def test_all_master_tickers_have_aliases(self):
        master_tickers = set(self.config.companies["ticker"].astype(str))
        alias_tickers = set(self.config.aliases["ticker"].astype(str))

        self.assertFalse(master_tickers - alias_tickers)

    def test_plural_keywords_match(self):
        result = self.classify("1 Megawatt Racks In Data Centers")

        self.assertIn("megawatt racks", result["matched_keywords"])
        self.assertEqual(result["filter_status"], "keep")

    def test_ucie_keyword_matches(self):
        result = self.classify("UCIe Interconnect Options For Chiplets")

        self.assertIn("UCIe", result["matched_keywords"])
        self.assertIn(result["filter_status"], {"keep", "review"})

    def test_data_movement_is_review_or_keep(self):
        result = self.classify("Overcoming Bottlenecks In Data Movement")

        self.assertIn("data movement", result["matched_keywords"])
        self.assertIn(result["filter_status"], {"keep", "review"})

    def test_week_in_review_is_roundup(self):
        self.assertEqual(
            classify_content("Week In Review: Manufacturing", ["News"], []),
            ("roundup", "B"),
        )

    def test_blog_review_is_roundup(self):
        self.assertEqual(
            classify_content("Blog Review: May 20", ["Top Stories"], []),
            ("roundup", "B"),
        )

    def test_technical_paper_roundup_is_roundup(self):
        self.assertEqual(
            classify_content("Chip Industry Technical Paper Roundup", ["Technical Papers"], []),
            ("roundup", "B"),
        )

    def test_research_bits_is_roundup(self):
        self.assertEqual(
            classify_content("Research Bits: Jun. 2", ["Top Stories"], []),
            ("roundup", "B"),
        )

    def test_regular_top_story_is_editorial(self):
        self.assertEqual(
            classify_content("Capacity Update", ["Top Stories"], []),
            ("editorial", "A"),
        )

    def test_content_taxonomy_classification(self):
        self.assertEqual(
            classify_content("New Paper", ["Technical Papers"], []),
            ("technical_paper", "B"),
        )
        self.assertEqual(
            classify_content("Vendor Guide", ["White Papers"], []),
            ("whitepaper", "C"),
        )


if __name__ == "__main__":
    unittest.main()

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from news import analyze_news_with_deepseek as cli
from news.deepseek_news_analyzer import (
    ANALYSIS_COLUMNS,
    normalize_analysis_result,
    parse_deepseek_json_response,
    summarize_deepseek_error,
)


CONFIG = {
    "api_key": "test-key",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-v4-flash",
    "prompt_version": "deepseek_news_v1",
    "analysis_limit": 20,
    "max_input_chars": 6000,
    "max_output_tokens": 800,
    "temperature": 0.1,
}


def article(news_id: str = "news_1") -> dict[str, str]:
    return {
        "news_id": news_id,
        "title": "AI networking demand grows",
        "url": "https://example.test/news",
        "published_at_gmt": "2026-06-28",
        "content_class": "editorial",
        "source_quality": "A",
        "matched_tickers": "ANET|MRVL",
        "matched_subthemes": "Networking|Optical",
        "matched_keywords": "networking|optical",
        "excerpt": "AI data center networking demand is growing.",
    }


def raw_result() -> dict:
    return {
        "relevance_score": 4,
        "impact_score": 3,
        "impact_direction": "positive",
        "signal_type": "demand",
        "recommended_decision": "keep",
        "primary_tickers": ["ANET", "MRVL"],
        "secondary_tickers": ["COHR", "LITE"],
        "primary_subthemes": ["Networking", "Optical"],
        "summary": "AI networking demand is improving.",
        "why_it_matters": "This supports the AI networking watchlist.",
        "follow_up_questions": ["What customers are buying?", "What changes in capex?"],
        "confidence": 4,
    }


def write_input(path: Path, rows: list[dict[str, str]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def args_for(tmp_path: Path, **overrides) -> argparse.Namespace:
    values = {
        "limit": 10,
        "dry_run": False,
        "force": False,
        "input_file": str(tmp_path / "review.csv"),
        "output_file": str(tmp_path / "analysis.csv"),
        "log_file": str(tmp_path / "analysis_log.csv"),
        "sleep_seconds": 0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class DeepSeekAnalyzerTests(unittest.TestCase):
    def test_parse_valid_json(self):
        parsed = parse_deepseek_json_response(json.dumps(raw_result()))

        self.assertEqual(parsed["recommended_decision"], "keep")
        self.assertEqual(parsed["primary_tickers"], ["ANET", "MRVL"])

    def test_parse_invalid_json(self):
        with self.assertRaisesRegex(ValueError, "invalid_json_response"):
            parse_deepseek_json_response("not json")

    def test_normalize_valid_result(self):
        normalized = normalize_analysis_result(
            article=article(),
            raw_result=raw_result(),
            config=CONFIG,
            input_chars=123,
            output_chars=456,
        )

        self.assertEqual(list(normalized.keys()), ANALYSIS_COLUMNS)
        self.assertEqual(normalized["primary_tickers"], "ANET,MRVL")
        self.assertEqual(normalized["secondary_tickers"], "COHR,LITE")
        self.assertEqual(normalized["follow_up_questions"], "What customers are buying? | What changes in capex?")
        self.assertEqual(normalized["relevance_score"], "4")

    def test_summarize_402_error(self):
        message = summarize_deepseek_error("402 Insufficient Balance")

        self.assertEqual(message, "DeepSeek API balance is insufficient")


class DeepSeekCliTests(unittest.TestCase):
    def test_skip_already_analyzed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_file = tmp_path / "review.csv"
            output_file = tmp_path / "analysis.csv"
            write_input(input_file, [article("news_1")])
            existing = normalize_analysis_result(
                article=article("news_1"),
                raw_result=raw_result(),
                config=CONFIG,
            )
            pd.DataFrame([existing], columns=ANALYSIS_COLUMNS).to_csv(
                output_file,
                index=False,
            )

            with patch.object(cli, "load_deepseek_config", return_value=CONFIG):
                with patch.object(cli, "load_deepseek_runtime_defaults", return_value=CONFIG):
                    with patch.object(cli, "analyze_article_with_deepseek") as analyze:
                        exit_code = cli.run_analysis(
                            args_for(
                                tmp_path,
                                input_file=str(input_file),
                                output_file=str(output_file),
                            )
                        )

            self.assertEqual(exit_code, 0)
            analyze.assert_not_called()
            output = pd.read_csv(output_file, dtype=str)
            self.assertEqual(len(output), 1)

    def test_force_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_file = tmp_path / "review.csv"
            output_file = tmp_path / "analysis.csv"
            write_input(input_file, [article("news_1")])
            old = normalize_analysis_result(
                article=article("news_1"),
                raw_result={**raw_result(), "summary": "old"},
                config=CONFIG,
            )
            new = normalize_analysis_result(
                article=article("news_1"),
                raw_result={**raw_result(), "summary": "new"},
                config=CONFIG,
            )
            pd.DataFrame([old], columns=ANALYSIS_COLUMNS).to_csv(output_file, index=False)

            with patch.object(cli, "load_deepseek_config", return_value=CONFIG):
                with patch.object(cli, "load_deepseek_runtime_defaults", return_value=CONFIG):
                    with patch.object(cli, "analyze_article_with_deepseek", return_value=new):
                        exit_code = cli.run_analysis(
                            args_for(
                                tmp_path,
                                force=True,
                                input_file=str(input_file),
                                output_file=str(output_file),
                            )
                        )

            output = pd.read_csv(output_file, dtype=str)
            self.assertEqual(exit_code, 0)
            self.assertEqual(len(output), 1)
            self.assertEqual(output.loc[0, "summary"], "new")

    def test_dry_run_does_not_call_api_or_write_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_file = tmp_path / "review.csv"
            output_file = tmp_path / "analysis.csv"
            write_input(input_file, [article("news_1")])

            with patch.object(cli, "load_deepseek_runtime_defaults", return_value=CONFIG):
                with patch.object(cli, "analyze_article_with_deepseek") as analyze:
                    exit_code = cli.run_analysis(
                        args_for(
                            tmp_path,
                            dry_run=True,
                            input_file=str(input_file),
                            output_file=str(output_file),
                        )
                    )

            self.assertEqual(exit_code, 0)
            analyze.assert_not_called()
            self.assertFalse(output_file.exists())

    def test_output_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_file = tmp_path / "review.csv"
            output_file = tmp_path / "analysis.csv"
            write_input(input_file, [article("news_1")])
            result = normalize_analysis_result(
                article=article("news_1"),
                raw_result=raw_result(),
                config=CONFIG,
            )

            with patch.object(cli, "load_deepseek_config", return_value=CONFIG):
                with patch.object(cli, "load_deepseek_runtime_defaults", return_value=CONFIG):
                    with patch.object(cli, "analyze_article_with_deepseek", return_value=result):
                        exit_code = cli.run_analysis(
                            args_for(
                                tmp_path,
                                input_file=str(input_file),
                                output_file=str(output_file),
                            )
                        )

            output = pd.read_csv(output_file, dtype=str)
            self.assertEqual(exit_code, 0)
            self.assertEqual(list(output.columns), ANALYSIS_COLUMNS)


if __name__ == "__main__":
    unittest.main()

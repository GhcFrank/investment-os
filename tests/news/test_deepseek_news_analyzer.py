import argparse
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from news import analyze_news_with_deepseek as cli
from news.deepseek_news_analyzer import (
    ANALYSIS_COLUMNS,
    analyze_article_with_deepseek,
    build_news_analysis_prompt,
    is_retryable_deepseek_error,
    load_company_universe,
    load_deepseek_config,
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
    "max_output_tokens": 1500,
    "temperature": 0.1,
    "max_retries": 2,
    "retry_sleep_seconds": 0,
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


def failed_result(news_id: str = "news_1") -> dict:
    return normalize_analysis_result(
        article=article(news_id),
        raw_result={},
        config=CONFIG,
        status="failed",
        error_message="empty_response",
        input_chars=100,
        output_chars=0,
    )


def response_with_content(content: str | None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
    )


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
        "news_ids_file": None,
        "news_id_column": "news_id",
        "skip_existing_attempts": False,
        "company_master_file": str(tmp_path / "company_master.csv"),
        "max_company_universe": 200,
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

    def test_load_company_universe_reads_company_master(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "company_master.csv"
            pd.DataFrame(
                [
                    {
                        "ticker": "anet",
                        "company": "Arista Networks",
                        "sector": "Technology",
                        "industry_group": "Networking Equipment",
                        "theme": "AI Infrastructure",
                        "subtheme": "Networking",
                        "supply_chain_layer": "Connectivity",
                        "business_quality_score": "8",
                    },
                    {
                        "ticker": "cohr",
                        "company": "Coherent",
                        "sector": "Technology",
                        "industry_group": "Optical Components",
                        "theme": "AI Infrastructure",
                        "subtheme": "Optical",
                        "supply_chain_layer": "Connectivity",
                        "business_quality_score": "7",
                    },
                ]
            ).to_csv(path, index=False)

            universe = load_company_universe(path)

        self.assertEqual(
            universe,
            [
                "ANET | Arista Networks | AI Infrastructure | Networking | Connectivity",
                "COHR | Coherent | AI Infrastructure | Optical | Connectivity",
            ],
        )

    def test_load_company_universe_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            universe = load_company_universe(Path(tmp) / "missing.csv")

        self.assertEqual(universe, [])

    def test_prompt_contains_company_universe(self):
        company_universe = [
            "ANET | Arista Networks | AI Infrastructure | Networking | Connectivity",
        ]

        messages = build_news_analysis_prompt(
            article=article(),
            company_universe=company_universe,
            max_input_chars=1000,
        )

        user_message = messages[1]["content"]
        self.assertIn("Supported company universe", user_message)
        self.assertIn("ANET | Arista Networks", user_message)

    def test_prompt_includes_json_instructions(self):
        messages = build_news_analysis_prompt(article=article(), max_input_chars=1000)
        combined = "\n".join(message["content"] for message in messages)

        self.assertIn("JSON", combined)
        self.assertIn("Return JSON only", combined)
        self.assertIn("Output a single valid JSON object", combined)
        self.assertIn("Do not wrap the JSON in Markdown", combined)
        self.assertIn("Do not include commentary outside the JSON object", combined)

    def test_prompt_does_not_truncate_json_schema(self):
        company_universe = [
            "ANET | Arista Networks | AI Infrastructure | Networking | Connectivity",
        ]
        long_article = {**article(), "excerpt": "x" * 20000}

        messages = build_news_analysis_prompt(
            article=long_article,
            company_universe=company_universe,
            max_input_chars=500,
        )

        user_message = messages[1]["content"]
        for expected in [
            "Return JSON only",
            "relevance_score",
            "impact_score",
            "recommended_decision",
            "primary_tickers",
            "follow_up_questions",
        ]:
            self.assertIn(expected, user_message)
        self.assertIn("[ARTICLE_CONTEXT_TRUNCATED]", user_message)

    def test_only_article_context_is_truncated(self):
        company_universe = [
            "ANET | Arista Networks | AI Infrastructure | Networking | Connectivity",
        ]
        long_article = {**article(), "excerpt": "x" * 20000}

        messages = build_news_analysis_prompt(
            article=long_article,
            company_universe=company_universe,
            max_input_chars=500,
        )

        self.assertIn("You are an investment research assistant", messages[0]["content"])
        self.assertIn("ANET | Arista Networks", messages[1]["content"])
        self.assertIn("JSON schema", messages[1]["content"])

    def test_empty_response_is_retryable(self):
        self.assertTrue(is_retryable_deepseek_error("empty_response"))

    def test_429_is_retryable(self):
        self.assertTrue(
            is_retryable_deepseek_error("DeepSeek API rate limit exceeded")
        )

    def test_402_is_not_retryable(self):
        self.assertFalse(
            is_retryable_deepseek_error("DeepSeek API balance is insufficient")
        )

    def test_analyze_retries_empty_response_then_succeeds(self):
        client = Mock()
        client.chat.completions.create.side_effect = [
            response_with_content(""),
            response_with_content(json.dumps(raw_result())),
        ]

        with patch("news.deepseek_news_analyzer.OpenAI", return_value=client):
            result = analyze_article_with_deepseek(
                article(),
                {**CONFIG, "max_retries": 2, "retry_sleep_seconds": 0},
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(client.chat.completions.create.call_count, 2)
        self.assertEqual(result["error_message"], "")

    def test_analyze_passes_json_response_format(self):
        company_universe = [
            "ANET | Arista Networks | AI Infrastructure | Networking | Connectivity",
        ]
        client = Mock()
        client.chat.completions.create.return_value = response_with_content(
            json.dumps(raw_result())
        )

        with patch("news.deepseek_news_analyzer.OpenAI", return_value=client):
            result = analyze_article_with_deepseek(
                article=article(),
                config=CONFIG,
                company_universe=company_universe,
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            client.chat.completions.create.call_args.kwargs["response_format"],
            {"type": "json_object"},
        )

    def test_default_max_output_tokens_is_1500(self):
        with patch("news.deepseek_news_analyzer.load_dotenv"):
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=True):
                config = load_deepseek_config()

        self.assertEqual(config["max_output_tokens"], 1500)

    def test_analyze_stops_after_max_retries(self):
        client = Mock()
        client.chat.completions.create.side_effect = [
            response_with_content(""),
            response_with_content(""),
            response_with_content(""),
        ]

        with patch("news.deepseek_news_analyzer.OpenAI", return_value=client):
            result = analyze_article_with_deepseek(
                article(),
                {**CONFIG, "max_retries": 2, "retry_sleep_seconds": 0},
            )

        self.assertEqual(client.chat.completions.create.call_count, 3)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_message"], "empty_response")

    def test_analyze_article_passes_company_universe_to_prompt_builder(self):
        company_universe = [
            "ANET | Arista Networks | AI Infrastructure | Networking | Connectivity",
        ]
        client = Mock()
        client.chat.completions.create.return_value = response_with_content(
            json.dumps(raw_result())
        )
        prompt_messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]

        with patch("news.deepseek_news_analyzer.OpenAI", return_value=client):
            with patch(
                "news.deepseek_news_analyzer.build_news_analysis_prompt",
                return_value=prompt_messages,
            ) as build_prompt:
                result = analyze_article_with_deepseek(
                    article=article(),
                    config=CONFIG,
                    company_universe=company_universe,
                )

        self.assertEqual(result["status"], "ok")
        build_prompt.assert_called_once_with(
            article=article(),
            company_universe=company_universe,
            max_input_chars=CONFIG["max_input_chars"],
        )


class DeepSeekCliTests(unittest.TestCase):
    def test_existing_ok_row_is_skipped_by_default(self):
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

    def test_existing_failed_row_is_retried_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_file = tmp_path / "review.csv"
            output_file = tmp_path / "analysis.csv"
            write_input(input_file, [article("news_1")])
            new = normalize_analysis_result(
                article=article("news_1"),
                raw_result=raw_result(),
                config=CONFIG,
            )
            pd.DataFrame([failed_result("news_1")], columns=ANALYSIS_COLUMNS).to_csv(
                output_file,
                index=False,
            )

            with patch.object(cli, "load_deepseek_config", return_value=CONFIG):
                with patch.object(cli, "load_deepseek_runtime_defaults", return_value=CONFIG):
                    with patch.object(cli, "analyze_article_with_deepseek", return_value=new) as analyze:
                        exit_code = cli.run_analysis(
                            args_for(
                                tmp_path,
                                input_file=str(input_file),
                                output_file=str(output_file),
                            )
                        )

            output = pd.read_csv(output_file, dtype=str)
            self.assertEqual(exit_code, 0)
            analyze.assert_called_once()
            self.assertEqual(len(output), 1)
            self.assertEqual(output.loc[0, "status"], "ok")

    def test_news_ids_file_allowlist_limits_articles(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_file = tmp_path / "review.csv"
            output_file = tmp_path / "analysis.csv"
            ids_file = tmp_path / "manifest.csv"
            write_input(input_file, [article("A"), article("B"), article("C")])
            pd.DataFrame([{"news_id": "B"}]).to_csv(ids_file, index=False)
            result = normalize_analysis_result(
                article=article("B"),
                raw_result=raw_result(),
                config=CONFIG,
            )

            with patch.object(cli, "load_deepseek_config", return_value=CONFIG):
                with patch.object(cli, "load_deepseek_runtime_defaults", return_value=CONFIG):
                    with patch.object(
                        cli,
                        "analyze_article_with_deepseek",
                        return_value=result,
                    ) as analyze:
                        exit_code = cli.run_analysis(
                            args_for(
                                tmp_path,
                                input_file=str(input_file),
                                output_file=str(output_file),
                                news_ids_file=str(ids_file),
                            )
                        )

            self.assertEqual(exit_code, 0)
            analyze.assert_called_once()
            self.assertEqual(analyze.call_args.kwargs["article"]["news_id"], "B")

    def test_empty_news_ids_file_exits_without_api_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_file = tmp_path / "review.csv"
            output_file = tmp_path / "analysis.csv"
            log_file = tmp_path / "analysis_log.csv"
            ids_file = tmp_path / "manifest.csv"
            write_input(input_file, [article("A")])
            pd.DataFrame(columns=["news_id"]).to_csv(ids_file, index=False)

            with patch.object(cli, "load_deepseek_runtime_defaults", return_value=CONFIG):
                with patch.object(cli, "load_deepseek_config") as load_config:
                    exit_code = cli.run_analysis(
                        args_for(
                            tmp_path,
                            input_file=str(input_file),
                            output_file=str(output_file),
                            log_file=str(log_file),
                            news_ids_file=str(ids_file),
                        )
                    )

            self.assertEqual(exit_code, 0)
            load_config.assert_not_called()
            log = pd.read_csv(log_file, dtype=str)
            self.assertEqual(log.loc[0, "status"], "skipped")
            self.assertEqual(log.loc[0, "articles_analyzed"], "0")

    def test_skip_existing_attempts_skips_failed_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_file = tmp_path / "review.csv"
            output_file = tmp_path / "analysis.csv"
            write_input(input_file, [article("B")])
            pd.DataFrame([failed_result("B")], columns=ANALYSIS_COLUMNS).to_csv(
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
                                skip_existing_attempts=True,
                            )
                        )

            self.assertEqual(exit_code, 0)
            analyze.assert_not_called()

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

    def test_force_retries_ok_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_file = tmp_path / "review.csv"
            output_file = tmp_path / "analysis.csv"
            write_input(input_file, [article("news_1")])
            old = normalize_analysis_result(
                article=article("news_1"),
                raw_result=raw_result(),
                config=CONFIG,
            )
            new = normalize_analysis_result(
                article=article("news_1"),
                raw_result={**raw_result(), "summary": "forced"},
                config=CONFIG,
            )
            pd.DataFrame([old], columns=ANALYSIS_COLUMNS).to_csv(output_file, index=False)

            with patch.object(cli, "load_deepseek_config", return_value=CONFIG):
                with patch.object(cli, "load_deepseek_runtime_defaults", return_value=CONFIG):
                    with patch.object(cli, "analyze_article_with_deepseek", return_value=new) as analyze:
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
            analyze.assert_called_once()
            self.assertEqual(len(output), 1)
            self.assertEqual(output.loc[0, "summary"], "forced")

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

    def test_dry_run_shows_previous_failed_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_file = tmp_path / "review.csv"
            output_file = tmp_path / "analysis.csv"
            write_input(input_file, [article("news_1"), article("news_2")])
            existing_ok = normalize_analysis_result(
                article=article("news_2"),
                raw_result=raw_result(),
                config=CONFIG,
            )
            pd.DataFrame(
                [failed_result("news_1"), existing_ok],
                columns=ANALYSIS_COLUMNS,
            ).to_csv(output_file, index=False)

            stdout = io.StringIO()
            with patch.object(cli, "load_deepseek_runtime_defaults", return_value=CONFIG):
                with patch.object(cli, "analyze_article_with_deepseek") as analyze:
                    with redirect_stdout(stdout):
                        exit_code = cli.run_analysis(
                            args_for(
                                tmp_path,
                                dry_run=True,
                                input_file=str(input_file),
                                output_file=str(output_file),
                            )
                        )

            output_text = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            analyze.assert_not_called()
            self.assertIn("To analyze: 1", output_text)
            self.assertIn("Skipped existing ok: 1", output_text)
            self.assertIn("Retrying previous failed: 1", output_text)
            self.assertIn("news_1 | previous failed", output_text)

    def test_dry_run_loads_company_universe(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_file = tmp_path / "review.csv"
            company_master_file = tmp_path / "company_master.csv"
            write_input(input_file, [article("news_1")])
            pd.DataFrame(
                [
                    {
                        "ticker": "ANET",
                        "company": "Arista Networks",
                        "theme": "AI Infrastructure",
                        "subtheme": "Networking",
                        "supply_chain_layer": "Connectivity",
                    },
                    {
                        "ticker": "COHR",
                        "company": "Coherent",
                        "theme": "AI Infrastructure",
                        "subtheme": "Optical",
                        "supply_chain_layer": "Connectivity",
                    },
                ]
            ).to_csv(company_master_file, index=False)

            stdout = io.StringIO()
            with patch.object(cli, "load_deepseek_runtime_defaults", return_value=CONFIG):
                with redirect_stdout(stdout):
                    exit_code = cli.run_analysis(
                        args_for(
                            tmp_path,
                            dry_run=True,
                            input_file=str(input_file),
                            company_master_file=str(company_master_file),
                        )
                    )

            self.assertEqual(exit_code, 0)
            self.assertIn("Loaded company universe: 2 companies", stdout.getvalue())

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

    def test_upsert_prevents_duplicate_key(self):
        old = failed_result("news_1")
        new = normalize_analysis_result(
            article=article("news_1"),
            raw_result=raw_result(),
            config=CONFIG,
        )

        output = cli.upsert_analysis_rows(
            pd.DataFrame([old], columns=ANALYSIS_COLUMNS),
            pd.DataFrame([new], columns=ANALYSIS_COLUMNS),
        )

        self.assertEqual(len(output), 1)
        self.assertEqual(output.loc[0, "status"], "ok")


if __name__ == "__main__":
    unittest.main()

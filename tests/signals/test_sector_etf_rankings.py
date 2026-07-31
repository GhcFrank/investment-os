import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from signals import build_sector_etf_metrics as metrics
from signals import build_sector_etf_rankings as rankings
from utils import send_email as email_utils


RANKING_DATE = "2025-01-02"
REFERENCE_DATES = {
    250: "2024-04-26",
    90: "2024-10-04",
    30: "2024-12-03",
}


def default_return_maps(config):
    tickers = [etf.ticker for etf in config.primary_sector_etfs]
    overlay_returns = {"SOXX": 0.035, "IGV": 0.045}
    return {
        250: {
            ticker: index / 100
            for index, ticker in enumerate(tickers)
        }
        | overlay_returns,
        90: {
            ticker: (len(tickers) - 1 - index) / 100
            for index, ticker in enumerate(tickers)
        }
        | overlay_returns,
        30: {
            ticker: value
            for ticker, value in zip(
                tickers,
                [0.03, 0.02, 0.01, 0.00, -0.01, -0.02,
                 -0.03, 0.04, 0.05, 0.06, 0.07],
            )
        }
        | {"SOXX": 0.015, "IGV": -0.005},
    }


def latest_metrics_fixture(
    *,
    missing_tickers=(),
    return_maps=None,
):
    config = rankings.load_sector_etf_config()
    values = return_maps or default_return_maps(config)
    rows = []
    for index, etf in enumerate(config.leadership_etfs):
        if etf.ticker in missing_tickers:
            continue
        adj_close = 100.0 + index
        row = {
            "date": RANKING_DATE,
            "ticker": etf.ticker,
            "sector_id": etf.sector_id,
            "sector_name": etf.sector_name,
            "sector_name_cn": etf.sector_name_cn,
            "adj_close": adj_close,
        }
        for horizon in metrics.SECTOR_ETF_RETURN_HORIZONS:
            return_value = values[horizon][etf.ticker]
            row[f"reference_date_{horizon}td"] = REFERENCE_DATES[horizon]
            row[f"reference_adj_close_{horizon}td"] = (
                adj_close / (1.0 + return_value)
                if not pd.isna(return_value)
                else np.nan
            )
            row[f"adj_close_return_{horizon}td"] = return_value
        rows.append(row)
    return rankings.LatestSectorETFMetrics(
        ranking_date=RANKING_DATE,
        rows=pd.DataFrame(rows),
        missing_tickers=tuple(sorted(missing_tickers)),
        configured_count=len(config.leadership_etfs),
        latest_dates={
            etf.ticker: (
                "2024-12-31"
                if etf.ticker in missing_tickers
                else RANKING_DATE
            )
            for etf in config.leadership_etfs
        },
    )


def set_horizon_returns(latest, horizon, values):
    rows = latest.rows.copy()
    for ticker, return_value in values.items():
        selected = rows["ticker"].eq(ticker)
        rows.loc[selected, f"adj_close_return_{horizon}td"] = return_value
        if pd.isna(return_value):
            rows.loc[
                selected,
                f"reference_adj_close_{horizon}td",
            ] = np.nan
        else:
            rows.loc[
                selected,
                f"reference_adj_close_{horizon}td",
            ] = (
                rows.loc[selected, "adj_close"] / (1.0 + return_value)
            )
    return replace(latest, rows=rows)


def write_metrics_files(
    directory,
    *,
    latest_date_by_ticker=None,
    omitted_tickers=(),
):
    config = metrics.load_sector_etf_config()
    latest_date_by_ticker = latest_date_by_ticker or {}
    output_dir = Path(directory)
    for index, etf in enumerate(config.leadership_etfs):
        if etf.ticker in omitted_tickers:
            continue
        latest_date = latest_date_by_ticker.get(etf.ticker, RANKING_DATE)
        dates = pd.bdate_range(end=latest_date, periods=251)
        prices = pd.DataFrame(
            {
                "date": dates.strftime("%Y-%m-%d"),
                "ticker": etf.ticker,
                "adj_close": np.linspace(
                    100.0 + index,
                    110.0 + index,
                    len(dates),
                ),
            }
        )
        built = metrics.build_one_sector_etf_metrics(prices)
        metrics.write_metrics_atomic(
            built,
            metrics.resolve_sector_etf_metrics_path(output_dir, etf),
        )
    return config


class SectorETFRankingCalculationTests(unittest.TestCase):
    def test_three_horizons_rank_independently_with_correct_directions(self):
        result = rankings.build_daily_sector_etf_rankings(
            latest_metrics_fixture()
        )

        def tickers(horizon, group):
            return list(
                result.loc[
                    result["horizon_trading_days"].eq(horizon)
                    & result["ranking_group"].eq(group)
                ].sort_values("rank")["ticker"]
            )

        self.assertEqual(tickers(250, "top"), ["XLU", "XLK", "XLRE"])
        self.assertEqual(tickers(250, "bottom"), ["XLC", "XLY", "XLP"])
        self.assertEqual(tickers(90, "top"), ["XLC", "XLY", "XLP"])
        self.assertEqual(tickers(90, "bottom"), ["XLU", "XLK", "XLRE"])
        self.assertEqual(tickers(30, "top"), ["XLU", "XLK", "XLRE"])
        self.assertEqual(tickers(30, "bottom"), ["XLI", "XLV", "XLF"])
        worst_30d = result.loc[
            result["horizon_trading_days"].eq(30)
            & result["ranking_group"].eq("bottom")
            & result["rank"].eq(1)
        ].iloc[0]
        self.assertEqual(worst_30d["ticker"], "XLI")
        self.assertAlmostEqual(worst_30d["adj_close_return"], -0.03)
        self.assertEqual(len(result), 18)

    def test_ties_use_ticker_ascending_and_numeric_strings_stay_numeric(self):
        config = rankings.load_sector_etf_config()
        values = default_return_maps(config)
        values[250].update(
            {
                "XLB": 0.50,
                "XLC": 0.50,
                "XLK": 0.50,
                "XLE": -0.50,
                "XLF": -0.50,
                "XLI": -0.50,
            }
        )
        latest = latest_metrics_fixture(return_maps=values)
        latest = replace(
            latest,
            rows=latest.rows.assign(
                adj_close_return_90td=latest.rows[
                    "adj_close_return_90td"
                ].map(str)
            ),
        )
        result = rankings.build_daily_sector_etf_rankings(latest)
        top = result.loc[
            result["horizon_trading_days"].eq(250)
            & result["ranking_group"].eq("top")
        ].sort_values("rank")
        bottom = result.loc[
            result["horizon_trading_days"].eq(250)
            & result["ranking_group"].eq("bottom")
        ].sort_values("rank")
        self.assertEqual(list(top["ticker"]), ["XLB", "XLC", "XLK"])
        self.assertEqual(list(bottom["ticker"]), ["XLE", "XLF", "XLI"])
        self.assertTrue(
            pd.api.types.is_numeric_dtype(result["adj_close_return"])
        )

    def test_null_return_is_excluded_not_treated_as_zero(self):
        latest = latest_metrics_fixture()
        latest = set_horizon_returns(latest, 250, {"XLU": np.nan})
        result = rankings.build_daily_sector_etf_rankings(latest)
        selected = result.loc[
            result["horizon_trading_days"].eq(250)
        ]
        self.assertNotIn("XLU", set(selected["ticker"]))
        self.assertEqual(set(selected["universe_size"]), {12})

    def test_fewer_than_six_valid_values_blocks_rankings(self):
        latest = latest_metrics_fixture()
        invalid_tickers = list(latest.rows["ticker"])[5:]
        latest = set_horizon_returns(
            latest,
            30,
            {ticker: np.nan for ticker in invalid_tickers},
        )
        with self.assertRaisesRegex(
            rankings.InsufficientRankingUniverseError,
            "only 5 valid ETF",
        ):
            rankings.build_daily_sector_etf_rankings(latest)


class SectorETFRankingDateTests(unittest.TestCase):
    def test_all_same_date_participate(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = write_metrics_files(tmp)
            latest = rankings.load_latest_sector_etf_metrics(
                config,
                metrics_dir=tmp,
            )
        self.assertEqual(latest.ranking_date, RANKING_DATE)
        self.assertEqual(latest.participating_count, 13)
        self.assertTrue(latest.is_complete)
        self.assertEqual(latest.missing_tickers, ())

    def test_missing_latest_date_is_marked_and_prior_row_is_not_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = write_metrics_files(
                tmp,
                latest_date_by_ticker={"XLF": "2024-12-31"},
            )
            latest = rankings.load_latest_sector_etf_metrics(
                config,
                metrics_dir=tmp,
            )
            built = rankings.build_daily_sector_etf_rankings(latest)
            email = rankings.format_sector_etf_ranking_email(
                built,
                participating_count=latest.participating_count,
                configured_count=latest.configured_count,
                missing_tickers=latest.missing_tickers,
            )

        self.assertEqual(latest.ranking_date, RANKING_DATE)
        self.assertIn("XLF", latest.missing_tickers)
        self.assertNotIn("XLF", set(latest.rows["ticker"]))
        self.assertEqual(latest.latest_dates["XLF"], "2024-12-31")
        self.assertTrue(email.incomplete)
        self.assertIn("[INCOMPLETE]", email.subject)
        self.assertIn("Missing Tickers: XLF", email.plain_text)

    def test_explicit_absent_date_is_not_shifted_to_another_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = write_metrics_files(tmp)
            with self.assertRaisesRegex(
                rankings.SectorETFRankingValidationError,
                "not present in metrics",
            ):
                rankings.load_latest_sector_etf_metrics(
                    config,
                    metrics_dir=tmp,
                    ranking_date="2025-01-04",
                )

    def test_five_available_etfs_do_not_produce_normal_rankings(self):
        config = rankings.load_sector_etf_config()
        omitted = {etf.ticker for etf in config.leadership_etfs[5:]}
        with tempfile.TemporaryDirectory() as tmp:
            config = write_metrics_files(tmp, omitted_tickers=omitted)
            latest = rankings.load_latest_sector_etf_metrics(
                config,
                metrics_dir=tmp,
            )
            with self.assertRaises(
                rankings.InsufficientRankingUniverseError
            ):
                rankings.build_daily_sector_etf_rankings(latest)


class SectorETFRankingHistoryTests(unittest.TestCase):
    def setUp(self):
        self.built = rankings.build_daily_sector_etf_rankings(
            latest_metrics_fixture()
        )

    def test_upsert_is_idempotent_revisable_and_appends_new_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            history_file = Path(tmp) / "rankings.csv"
            self.assertTrue(
                rankings.upsert_sector_etf_ranking_history(
                    self.built,
                    history_file,
                )
            )
            first_mtime = history_file.stat().st_mtime_ns
            self.assertFalse(
                rankings.upsert_sector_etf_ranking_history(
                    self.built,
                    history_file,
                )
            )
            self.assertEqual(first_mtime, history_file.stat().st_mtime_ns)

            revised = self.built.copy()
            revised.loc[0, "adj_close"] *= 2
            revised.loc[0, "reference_adj_close"] *= 2
            self.assertTrue(
                rankings.upsert_sector_etf_ranking_history(
                    revised,
                    history_file,
                )
            )
            after_revision = rankings.load_sector_etf_ranking_history(
                history_file
            )
            self.assertEqual(len(after_revision), 18)
            self.assertFalse(
                after_revision.duplicated(
                    [
                        "date",
                        "horizon_trading_days",
                        "ranking_group",
                        "rank",
                    ]
                ).any()
            )

            next_date = revised.copy()
            next_date["date"] = "2025-01-03"
            self.assertTrue(
                rankings.upsert_sector_etf_ranking_history(
                    next_date,
                    history_file,
                )
            )
            saved = rankings.load_sector_etf_ranking_history(history_file)

        self.assertEqual(len(saved), 36)
        self.assertEqual(
            list(saved["date"].drop_duplicates()),
            ["2025-01-02", "2025-01-03"],
        )
        first_date = saved.loc[saved["date"].eq("2025-01-02")]
        self.assertEqual(
            list(first_date["horizon_trading_days"].drop_duplicates()),
            [250, 90, 30],
        )
        first_horizon = first_date.loc[
            first_date["horizon_trading_days"].eq(250)
        ]
        self.assertEqual(
            list(first_horizon["ranking_group"].drop_duplicates()),
            ["top", "bottom"],
        )

    def test_output_schema_is_exact_long_format(self):
        self.assertEqual(
            list(self.built.columns),
            rankings.RANKING_COLUMNS,
        )
        self.assertEqual(
            len(self.built),
            3 * 2 * 3,
        )

    def test_full_history_rebuild_replaces_old_data_without_email(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_dir = root / "metrics"
            write_metrics_files(metrics_dir)
            history_file = root / "rankings.csv"
            history_file.write_text(
                "old_calendar_day_history\n",
                encoding="utf-8",
            )
            email_log = root / "email_log.csv"
            email_log.write_text(
                "ranking_date,sent_at_utc,status,recipient_count,"
                "error_message\n"
                "2025-01-01,2025-01-01T22:00:00Z,success,1,\n",
                encoding="utf-8",
            )
            email_log_before = email_log.read_bytes()

            with patch.object(
                rankings,
                "send_sector_etf_ranking_email",
            ) as sender:
                first = rankings.rebuild_sector_etf_ranking_history(
                    metrics_dir=metrics_dir,
                    ranking_history_file=history_file,
                )
                first_bytes = history_file.read_bytes()
                second = rankings.rebuild_sector_etf_ranking_history(
                    metrics_dir=metrics_dir,
                    ranking_history_file=history_file,
                )

            saved = rankings.load_sector_etf_ranking_history(history_file)
            second_bytes = history_file.read_bytes()
            email_log_after = email_log.read_bytes()

        sender.assert_not_called()
        self.assertEqual(first.configured_etfs, 13)
        self.assertEqual(first.common_dates, 251)
        self.assertEqual(first.ranked_dates, 1)
        self.assertEqual(first.skipped_unrankable_dates, 250)
        self.assertEqual(first.ranking_rows, 18)
        self.assertTrue(first.history_written)
        self.assertFalse(second.history_written)
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(email_log_before, email_log_after)
        self.assertEqual(list(saved.columns), rankings.RANKING_COLUMNS)
        self.assertFalse(
            saved.duplicated(
                [
                    "date",
                    "horizon_trading_days",
                    "ranking_group",
                    "rank",
                ]
            ).any()
        )
        self.assertEqual(saved.groupby("date").size().tolist(), [18])


class SectorETFRankingEmailTests(unittest.TestCase):
    def setUp(self):
        self.built = rankings.build_daily_sector_etf_rankings(
            latest_metrics_fixture()
        )
        self.email = rankings.format_sector_etf_ranking_email(
            self.built,
            participating_count=13,
            configured_count=13,
        )

    def test_plain_and_html_templates_have_all_required_sections(self):
        self.assertEqual(
            self.email.subject,
            "[Investment OS] Sector ETF Rotation Rankings - 2025-01-02",
        )
        self.assertFalse(self.email.incomplete)
        self.assertEqual(rankings.EMAIL_HORIZON_ORDER, (30, 90, 250))
        for horizon in rankings.EMAIL_HORIZON_ORDER:
            self.assertIn(
                f"{horizon}-Trading-Day Return",
                self.email.plain_text,
            )
            self.assertIn(
                f"{horizon}-Trading-Day Return",
                self.email.html,
            )
        for content in (self.email.plain_text, self.email.html):
            self.assertLess(
                content.index("30-Trading-Day Return"),
                content.index("90-Trading-Day Return"),
            )
            self.assertLess(
                content.index("90-Trading-Day Return"),
                content.index("250-Trading-Day Return"),
            )
            self.assertIn("fixed trading-session lookback", content)
            self.assertNotIn("calendar-day", content.casefold())
        self.assertEqual(self.email.plain_text.count("Top 3"), 3)
        self.assertEqual(self.email.plain_text.count("Bottom 3"), 3)
        self.assertEqual(self.email.html.count("<table>"), 6)
        self.assertIn("10.00%", self.email.plain_text)
        self.assertIn("Reference Date", self.email.plain_text)
        self.assertIn("Adjusted Close", self.email.plain_text)
        self.assertIn(
            "Price Source: Yahoo Finance Adjusted Close",
            self.email.plain_text,
        )
        self.assertIn("Leadership Universe Size: 13", self.email.plain_text)
        self.assertIn("Participating ETFs: 13/13", self.email.plain_text)
        self.assertIn(
            "11 primary sectors + SOXX semiconductors + IGV software",
            self.email.plain_text,
        )
        self.assertTrue(
            pd.api.types.is_numeric_dtype(self.built["adj_close_return"])
        )

    def test_each_email_horizon_keeps_top_then_bottom_three_unchanged(self):
        before = self.built.copy(deep=True)

        for index, horizon in enumerate(rankings.EMAIL_HORIZON_ORDER):
            heading = f"{horizon}-Trading-Day Return"
            next_heading = (
                f"{rankings.EMAIL_HORIZON_ORDER[index + 1]}"
                "-Trading-Day Return"
                if index + 1 < len(rankings.EMAIL_HORIZON_ORDER)
                else None
            )
            plain_start = self.email.plain_text.index(heading)
            html_start = self.email.html.index(heading)
            plain_end = (
                self.email.plain_text.index(next_heading)
                if next_heading
                else len(self.email.plain_text)
            )
            html_end = (
                self.email.html.index(next_heading)
                if next_heading
                else len(self.email.html)
            )
            plain_section = self.email.plain_text[plain_start:plain_end]
            html_section = self.email.html[html_start:html_end]

            self.assertLess(
                plain_section.index("Top 3"),
                plain_section.index("Bottom 3"),
            )
            self.assertLess(
                html_section.index("Top 3"),
                html_section.index("Bottom 3"),
            )
            plain_groups = {
                "top": plain_section[
                    plain_section.index("Top 3"):
                    plain_section.index("Bottom 3")
                ],
                "bottom": plain_section[
                    plain_section.index("Bottom 3"):
                ],
            }
            html_groups = {
                "top": html_section[
                    html_section.index("Top 3"):
                    html_section.index("Bottom 3")
                ],
                "bottom": html_section[
                    html_section.index("Bottom 3"):
                ],
            }
            for ranking_group in rankings.RANKING_GROUP_ORDER:
                selected = self.built.loc[
                    self.built["horizon_trading_days"].eq(horizon)
                    & self.built["ranking_group"].eq(ranking_group)
                ].sort_values("rank")
                self.assertEqual(len(selected), 3)
                for row in selected.itertuples():
                    percentage = f"{row.adj_close_return:.2%}"
                    self.assertIn(
                        f"{row.rank} | {row.ticker} |",
                        plain_groups[ranking_group],
                    )
                    self.assertIn(
                        f"<td>{row.ticker}</td>",
                        html_groups[ranking_group],
                    )
                    self.assertIn(
                        percentage,
                        plain_groups[ranking_group],
                    )
                    self.assertIn(
                        percentage,
                        html_groups[ranking_group],
                    )

        pd.testing.assert_frame_equal(self.built, before)

    def test_test_email_wrapper_adds_markers_and_preserves_formal_content(self):
        test_email = rankings.format_sector_etf_test_email(
            self.email,
            ranking_date=RANKING_DATE,
        )
        self.assertEqual(
            test_email.subject,
            (
                "[TEST][Investment OS] Sector ETF Leadership Rankings - "
                "2025-01-02"
            ),
        )
        for content in (test_email.plain_text, test_email.html):
            self.assertIn(
                "TEST EMAIL — Format validation only.",
                content,
            )
            self.assertLess(
                content.index("30-Trading-Day Return"),
                content.index("90-Trading-Day Return"),
            )
            self.assertLess(
                content.index("90-Trading-Day Return"),
                content.index("250-Trading-Day Return"),
            )
            self.assertEqual(content.count("Top 3"), 3)
            self.assertEqual(content.count("Bottom 3"), 3)
        rankings.validate_sector_etf_test_email_preview(
            test_email,
            self.built,
            ranking_date=RANKING_DATE,
        )

        incomplete = replace(self.email, incomplete=True)
        incomplete_test = rankings.format_sector_etf_test_email(
            incomplete,
            ranking_date=RANKING_DATE,
        )
        self.assertTrue(
            incomplete_test.subject.startswith(
                "[TEST][INCOMPLETE][Investment OS]"
            )
        )

    def test_real_test_mode_uses_renderer_once_without_mutating_data_or_log(
        self,
    ):
        calls = []

        def mock_sender(**kwargs):
            calls.append(kwargs)
            return 1

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_dir = root / "metrics"
            config = write_metrics_files(metrics_dir)
            latest = rankings.load_latest_sector_etf_metrics(
                config,
                metrics_dir=metrics_dir,
            )
            built = rankings.build_daily_sector_etf_rankings(latest)
            history_file = root / "rankings.csv"
            rankings.upsert_sector_etf_ranking_history(
                built,
                history_file,
            )
            history_before = history_file.read_bytes()
            metrics_before = {
                path.name: path.read_bytes()
                for path in metrics_dir.glob("*.csv")
            }

            real_renderer = rankings.format_sector_etf_ranking_email
            with (
                patch.object(
                    rankings,
                    "format_sector_etf_ranking_email",
                    wraps=real_renderer,
                ) as renderer,
                patch.object(
                    rankings,
                    "_upsert_sector_etf_email_log",
                ) as production_log_writer,
            ):
                result = rankings.run_sector_etf_test_email(
                    metrics_dir=metrics_dir,
                    ranking_history_file=history_file,
                    email_sender=mock_sender,
                )

            history_after = history_file.read_bytes()
            metrics_after = {
                path.name: path.read_bytes()
                for path in metrics_dir.glob("*.csv")
            }

        renderer.assert_called_once()
        production_log_writer.assert_not_called()
        self.assertEqual(len(calls), 1)
        self.assertIn("html_body", calls[0])
        self.assertTrue(calls[0]["subject"].startswith("[TEST]"))
        self.assertEqual(result.recipient_count, 1)
        self.assertEqual(history_before, history_after)
        self.assertEqual(metrics_before, metrics_after)

    def test_test_email_and_force_email_are_rejected(self):
        with self.assertRaises(SystemExit):
            rankings.parse_args(["--test-email", "--force-email"])

    def test_success_duplicate_force_error_and_retry_are_idempotent(self):
        calls = []

        def successful_sender(**kwargs):
            calls.append(kwargs)
            return 2

        with tempfile.TemporaryDirectory() as tmp:
            log_file = Path(tmp) / "email_log.csv"
            first = rankings.send_sector_etf_ranking_email(
                self.email,
                ranking_date=RANKING_DATE,
                email_log_file=log_file,
                email_sender=successful_sender,
                now_utc=lambda: "2025-01-02T22:00:00Z",
            )
            duplicate = rankings.send_sector_etf_ranking_email(
                self.email,
                ranking_date=RANKING_DATE,
                email_log_file=log_file,
                email_sender=successful_sender,
                now_utc=lambda: "2025-01-03T22:00:00Z",
            )
            forced = rankings.send_sector_etf_ranking_email(
                self.email,
                ranking_date=RANKING_DATE,
                email_log_file=log_file,
                force_email=True,
                email_sender=successful_sender,
                now_utc=lambda: "2025-01-03T22:00:00Z",
            )
            saved = rankings.load_sector_etf_email_log(log_file)

        self.assertEqual(first.status, "success")
        self.assertEqual(duplicate.status, "skipped_duplicate")
        self.assertEqual(forced.status, "success")
        self.assertEqual(len(calls), 2)
        self.assertIn("html_body", calls[0])
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved.iloc[0]["status"], "success")
        self.assertEqual(saved.iloc[0]["recipient_count"], 2)

        attempts = []

        def retrying_sender(**kwargs):
            attempts.append(kwargs)
            if len(attempts) == 1:
                raise RuntimeError("SMTP unavailable")
            return 1

        with tempfile.TemporaryDirectory() as tmp:
            log_file = Path(tmp) / "email_log.csv"
            failed = rankings.send_sector_etf_ranking_email(
                self.email,
                ranking_date=RANKING_DATE,
                email_log_file=log_file,
                email_sender=retrying_sender,
                now_utc=lambda: "2025-01-02T22:00:00Z",
            )
            failed_log = rankings.load_sector_etf_email_log(log_file)
            retried = rankings.send_sector_etf_ranking_email(
                self.email,
                ranking_date=RANKING_DATE,
                email_log_file=log_file,
                email_sender=retrying_sender,
                now_utc=lambda: "2025-01-02T22:05:00Z",
            )
            success_log = rankings.load_sector_etf_email_log(log_file)

        self.assertEqual(failed.status, "error")
        self.assertEqual(failed_log.iloc[0]["status"], "error")
        self.assertEqual(retried.status, "success")
        self.assertEqual(success_log.iloc[0]["status"], "success")
        self.assertEqual(len(attempts), 2)

    def test_email_failure_keeps_saved_ranking_history(self):
        def failing_sender(**kwargs):
            raise RuntimeError("SMTP unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_dir = root / "metrics"
            write_metrics_files(metrics_dir)
            history_file = root / "rankings.csv"
            log_file = root / "email_log.csv"
            summary = rankings.run_sector_etf_daily_ranking(
                metrics_dir=metrics_dir,
                ranking_history_file=history_file,
                email_log_file=log_file,
                send_email_message=True,
                email_sender=failing_sender,
            )
            saved = rankings.load_sector_etf_ranking_history(history_file)
            log = rankings.load_sector_etf_email_log(log_file)

        self.assertEqual(summary.email_status, "error")
        self.assertEqual(len(saved), 18)
        self.assertEqual(log.iloc[0]["status"], "error")

    def test_existing_sender_builds_multipart_without_real_smtp(self):
        smtp = MagicMock()
        smtp.__enter__.return_value = smtp
        with (
            patch.dict(
                os.environ,
                {
                    "GMAIL_USER": "sender@example.com",
                    "GMAIL_APP_PASSWORD": "test-password",
                    "EMAIL_TO": "one@example.com,two@example.com",
                },
                clear=False,
            ),
            patch.object(email_utils, "load_dotenv"),
            patch.object(email_utils.smtplib, "SMTP_SSL", return_value=smtp),
        ):
            recipient_count = email_utils.send_email(
                subject="Test",
                body="Plain",
                html_body="<p>HTML</p>",
            )

        self.assertEqual(recipient_count, 2)
        smtp.login.assert_called_once_with(
            "sender@example.com",
            "test-password",
        )
        sent_recipients = smtp.sendmail.call_args.args[1]
        raw_message = smtp.sendmail.call_args.args[2]
        self.assertEqual(
            sent_recipients,
            ["one@example.com", "two@example.com"],
        )
        self.assertIn("multipart/alternative", raw_message)
        self.assertIn("text/plain", raw_message)
        self.assertIn("text/html", raw_message)


class SectorETFRankingRunTests(unittest.TestCase):
    def test_local_run_writes_18_rows_and_second_run_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_dir = root / "metrics"
            write_metrics_files(metrics_dir)
            history_file = root / "rankings.csv"
            first = rankings.run_sector_etf_daily_ranking(
                metrics_dir=metrics_dir,
                ranking_history_file=history_file,
                email_log_file=root / "email_log.csv",
            )
            first_mtime = history_file.stat().st_mtime_ns
            second = rankings.run_sector_etf_daily_ranking(
                metrics_dir=metrics_dir,
                ranking_history_file=history_file,
                email_log_file=root / "email_log.csv",
            )
            second_mtime = history_file.stat().st_mtime_ns

        self.assertEqual(first.ranking_date, RANKING_DATE)
        self.assertEqual(first.participating_etfs, 13)
        self.assertEqual(first.ranking_rows, 18)
        self.assertTrue(first.history_written)
        self.assertEqual(first.email_status, "not_requested")
        self.assertFalse(second.history_written)
        self.assertEqual(first_mtime, second_mtime)


if __name__ == "__main__":
    unittest.main()

# Investment OS

Personal market research automation project.

## Layout

- `src/pipelines/`: pipeline entrypoints
- `src/market_data/`: prices, fundamentals, ranking, history utilities
- `src/signals/`: sector and market signal generation
- `src/events/`: earnings calendar and IR event monitors
- `src/prediction_markets/`: Polymarket ingestion, matching, snapshots, alerts
- `src/utils/`: shared utilities such as email delivery
- `data/master/`: manually maintained master data
- `data/market_data/`: generated price/fundamental/ranking datasets
- `data/signals/`: generated signal datasets
- `data/events/`: generated event/calendar datasets
- `data/prediction_markets/`: generated prediction market datasets

Run the daily pipeline:

```bash
PYTHONPATH=src python -m pipelines.run_daily_pipeline
```

## Sector and industry ETF market and fund data

The market-data layer tracks one Select Sector SPDR ETF for each of the 11
GICS top-level US equity sectors plus two industry overlays: SOXX
(semiconductors) and IGV (software). `config/sector_etfs.yaml` explicitly
defines two non-interchangeable universes:

- `primary_sector`: the original 11 GICS sector ETFs only. Any top-level
  sector concentration or breadth calculation must use this universe.
- `leadership`: the 11 primary-sector ETFs plus SOXX and IGV (13 total), used
  for Yahoo prices, adjusted-close metrics, rankings, and ranking email.

SOXX and IGV are industry overlays and are not additional GICS top-level
sectors.

The providers have separate responsibilities:

- Yahoo Finance/yfinance writes market OHLCV history for all 13 leadership
  ETFs to
  `data/market_data/sector_etf_prices.csv`. `close` is the raw exchange close;
  `adj_close` includes company-action adjustments.
- State Street's official NAV History workbooks write one raw fund-history CSV
  for each of the original 11 primary-sector ETFs under
  `data/market_data/sector_etf_fund_history/`. Output basenames
  come from each ETF's `fund_history_filename` in `config/sector_etfs.yaml`;
  examples include `xlc_communication_services.csv`, `xlf_finance.csv`,
  `xlk_information_technology.csv`, and `xlre_real_estate.csv`. Each file
  contains `date`, `nav`, `shares_outstanding`, and `total_net_assets`.
- The official iShares/BlackRock fund download supplies historical daily NAV
  and shares for SOXX and IGV. The official product page supplies the latest
  dated Net Assets of Fund. That official AUM is stored only on its matching
  date; historical AUM is left blank because the download does not provide it.
  The output files are `soxx_semiconductors.csv` and `igv_software.csv`.

NAV is the fund's per-share net asset value and is not the exchange market
close. `total_net_assets` is the official fund AUM. Early records can contain
NAV while shares and AUM are blank; missing values are never filled with zero.
Downloaded XLSX or SpreadsheetML files are content-validated, parsed in
memory, and are not retained. `NAV × shares` is only a consistency check and
never substitutes for official AUM.

Bootstrap all official fund histories:

```bash
python -m src.market_data.update_sector_etf_fund_history --bootstrap
```

Daily/idempotent update, or a selected subset:

```bash
python -m src.market_data.update_sector_etf_fund_history
python -m src.market_data.update_sector_etf_fund_history \
  --tickers XLF,XLC,XLRE
```

Update both iShares industry overlays, or one selected ticker:

```bash
python -m src.market_data.update_ishares_etf_fund_history
python -m src.market_data.update_ishares_etf_fund_history --tickers SOXX
```

Update Yahoo market prices separately:

```bash
python -m src.market_data.update_sector_etf_prices
```

The daily pipeline runs both provider-specific fund steps before Yahoo price
and signal generation. The obsolete Yahoo AUM snapshot remains removed;
Yahoo fund metadata is never used as official AUM.

Yahoo, State Street, and iShares/BlackRock data can be delayed, unavailable,
or revised. Their data quality and availability are not guaranteed by this
project. The raw fund CSVs do not contain inferred flows, returns, rankings,
or rotation signals.

## Sector ETF adjusted-close return metrics

The local-only metrics builder reads
`data/market_data/sector_etf_prices.csv` after its Yahoo price update and writes
one full-history CSV per leadership ETF under
`data/signals/sector_etf_metrics/`. It reuses each ETF's configured
`metrics_filename`, including `xlf_finance.csv`,
`soxx_semiconductors.csv`, and `igv_software.csv`; it does not create
ticker-only files.

Each output has this fixed schema:

```text
date
adj_close
reference_date_250td
reference_adj_close_250td
adj_close_return_250td
reference_date_90td
reference_adj_close_90td
adj_close_return_90td
reference_date_30td
reference_adj_close_30td
adj_close_return_30td
```

The 250, 90, and 30 horizons are fixed trading-session counts in each ETF's
own local Yahoo price history. They do not use calendar-day or market-calendar
approximations. For a row at index `i`, the reference row is exactly
`i - horizon`, equivalent to `shift(horizon)`. The calculations are:

```text
30-trading-day return =
current adjusted close / adjusted close 30 ETF trading observations earlier - 1

90-trading-day return =
current adjusted close / adjusted close 90 ETF trading observations earlier - 1

250-trading-day return =
current adjusted close / adjusted close 250 ETF trading observations earlier - 1
```

Thus `0.25` means a 25% gain and `-0.20` means a 20% loss. The builder uses
Yahoo market `adj_close` exclusively—not raw `close` and not State Street
or iShares NAV—and it does not forward-fill missing observations or infer
sessions from a market calendar. If an ETF does not yet have enough history,
all three fields for that horizon remain empty while the current row is
retained.

Rebuild the complete local history (the default command has the same
full-rebuild behavior):

```bash
python -m src.signals.build_sector_etf_metrics --rebuild
```

The daily pipeline invokes this builder in-process after the Yahoo sector ETF
price update. It rewrites a metrics file
only when its canonical content changes. The directory is intended to support
future sector-rotation fields such as relative strength, volume, flows, and
rankings without changing the raw market or fund-history datasets.

## Sector ETF daily rankings and email

The sector ETF portion of the daily pipeline runs in this strict order:

```text
1. State Street fund history update
2. iShares SOXX/IGV fund history update
3. Yahoo Finance price update for all 13 leadership ETFs
4. 30/90/250 trading-day adjusted-close return update for all 13
5. Daily 13-ETF leadership Top 3/Bottom 3 ranking and email
```

After all metrics files are updated, the ranking builder reads the same exact
trading date from every ETF in the explicit 13-member leadership universe. It
never substitutes a ticker's prior
date when current-date data is missing. Each horizon is ranked independently
by numeric return, with ticker ascending as the deterministic tie-break.
Missing returns do not participate.

The long-format ranking history is stored at:

```text
data/signals/sector_etf_daily_rankings.csv
```

Each normal trading date still contributes 18 rows: three horizons, Top 3 and
Bottom 3, and three ranks in each group. `horizon_trading_days` stores the
30, 90, or 250 trading-observation lookback, and `universe_size` is normally
13. Re-running a date replaces that date's rows, allowing revised Yahoo
adjusted-close history to revise the ranking without creating duplicate keys.

Replace the complete historical ranking file from the 13 local metrics files,
without sending email or changing the production email log:

```bash
python -m src.signals.build_sector_etf_rankings --rebuild-history
```

Build and save the latest ranking without sending email:

```bash
python -m src.signals.build_sector_etf_rankings
```

Render the plain-text and HTML alternatives without sending:

```bash
python -m src.signals.build_sector_etf_rankings --dry-run-email
```

The daily pipeline uses the existing `GMAIL_USER`, `GMAIL_APP_PASSWORD`, and
`EMAIL_TO` settings to send one ranking email per successful ranking date. The
email contains the 30-, 90-, and 250-trading-day Top 3 and Bottom 3 tables, in
that presentation order. Its header describes an ETF leadership universe
rather than incorrectly calling all 13 entries primary sectors. Weekend,
holiday, and retry runs use the last market-data date and do not resend after
that ranking date has a successful log entry. Failed sends are recorded and
may be retried; `--send-email --force-email` explicitly resends a successful
date.

Email status is stored at:

```text
data/signals/sector_etf_ranking_email_log.csv
```

Returns in both the metrics and ranking CSVs remain decimal values. Email
templates render them as percentages and state that the rankings are a
quantitative summary, not an investment recommendation.

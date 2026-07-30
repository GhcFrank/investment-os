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
PYTHONPATH=src python src/pipelines/run_daily_pipeline.py
```

## GICS sector ETF market and fund data

The market-data layer tracks one Select Sector SPDR ETF for each of the 11
GICS top-level US equity sectors. The single ticker/sector mapping and State
Street URL template are in `config/sector_etfs.yaml`.

The providers have separate responsibilities:

- Yahoo Finance/yfinance writes market OHLCV history to
  `data/market_data/sector_etf_prices.csv`. `close` is the raw exchange close;
  `adj_close` includes company-action adjustments.
- State Street's official NAV History workbooks write one raw fund-history CSV
  per ETF under `data/market_data/sector_etf_fund_history/`. Output basenames
  come from each ETF's `fund_history_filename` in `config/sector_etfs.yaml`;
  examples include `xlc_communication_services.csv`, `xlf_finance.csv`,
  `xlk_information_technology.csv`, and `xlre_real_estate.csv`. Each file
  contains `date`, `nav`, `shares_outstanding`, and `total_net_assets`.

NAV is the fund's per-share net asset value and is not the exchange market
close. `total_net_assets` is the official fund AUM. Early records can contain
NAV while shares and AUM are blank; missing values are never filled with zero.
Downloaded XLSX files are parsed in memory and are not retained.

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

Update Yahoo market prices separately:

```bash
python -m src.market_data.update_sector_etf_prices
```

The daily pipeline runs both provider-specific steps automatically before
signal generation. The obsolete Yahoo AUM snapshot was removed; official
historical NAV, shares, and AUM now come exclusively from State Street.

Yahoo and State Street data can be delayed, unavailable, or revised. Their data
quality and availability are not guaranteed by this project. The raw fund CSVs
do not contain inferred flows, returns, rankings, or rotation signals.

## Sector ETF adjusted-close return metrics

The local-only metrics builder reads
`data/market_data/sector_etf_prices.csv` after its Yahoo price update and writes
one full-history CSV per configured ETF under
`data/signals/sector_etf_metrics/`. It reuses each ETF's configured
sector-specific filename, including `xlf_finance.csv`; it does not create
ticker-only files.

Each output has this fixed schema:

```text
date
adj_close
reference_date_250d
reference_adj_close_250d
adj_close_return_250d
reference_date_90d
reference_adj_close_90d
adj_close_return_90d
reference_date_30d
reference_adj_close_30d
adj_close_return_30d
```

The 250, 90, and 30 horizons are calendar days, not trading-session row
counts. For each current trading date, the reference is the latest trading
date on or before the calendar-day target. The calculation is:

```text
current_adj_close / reference_adj_close - 1
```

Thus `0.25` means a 25% gain and `-0.20` means a 20% loss. The builder uses
Yahoo market `adj_close` exclusively—not raw `close` and not State Street
NAV—so distributions, splits, and other applicable company actions are
reflected. If an ETF does not yet have enough history, all three fields for
that horizon remain empty while the current row is retained.

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
2. Yahoo Finance price update
3. 250/90/30 adjusted-close return update
4. Daily Top 3/Bottom 3 ranking and email
```

After all metrics files are updated, the ranking builder reads the same exact
trading date from every configured ETF. It never substitutes a ticker's prior
date when current-date data is missing. Each horizon is ranked independently
by numeric return, with ticker ascending as the deterministic tie-break.
Missing returns do not participate.

The long-format ranking history is stored at:

```text
data/signals/sector_etf_daily_rankings.csv
```

Each normal trading date contributes 18 rows: three horizons, Top 3 and Bottom
3, and three ranks in each group. Re-running a date replaces that date's rows,
allowing revised Yahoo adjusted-close history to revise the ranking without
creating duplicate keys.

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
email contains the 250-, 90-, and 30-day Top 3 and Bottom 3 tables. Weekend,
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

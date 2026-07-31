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

The GitHub Actions daily job injects `VAST_API_KEY` from the repository secret
of the same name. The key is never stored in source, workflow YAML, generated
CSV files, or email content.

## VIX market sentiment and CNN daily status

The daily pipeline updates VIX from Yahoo Finance through a VIX-only production
entrypoint, then writes the formal current signal to
`data/signals/vix_market_sentiment.csv` before any email is rendered. VIX does
not require a separate API key. Its unified-email section contains the VIX
level, 1-, 5-, and 20-observation point changes, sentiment regime, conservative
interpretation, source data date, data status, and an explicit stale flag.
Changes are VIX index-point changes over the existing daily market observations,
not percentages.

The VIX signal uses `SUCCESS`, `SUCCESS_WITH_WARNINGS`,
`INSUFFICIENT_HISTORY`, and `DATA_UNAVAILABLE`. Missing change history is shown
as `Insufficient history`, never zero, `nan`, or `None`. A failed update does
not reuse the prior row silently: the daily pipeline continues and the unified
email displays `VIX market sentiment data unavailable`. VIX has no separate
production email; its section is rendered before the ETF leadership and GPU
sections in the one unified daily message.

CNN Fear & Greed is temporarily disabled in the daily pipeline and GitHub
Actions because its current endpoint is not reliable. Daily runs do not request
CNN, update its current/history CSV files, render a CNN email section, or use
`SENTIMENT_USER_AGENT`. The CNN fetch/parser, fixtures, tests, historical CSVs,
and manual module entrypoint remain intact for future repair:

```bash
PYTHONPATH=src python -m market_data.update_sentiment_indicators --dry-run
```

Re-enabling CNN requires an explicit future pipeline change after the manual
path is fixed and validated; it is not controlled by a hidden default-on flag.
The unified email still omits the `Generated Files` section while all enabled
daily outputs continue to be saved.

## Vast.ai GPU cloud market snapshots

The GPU cloud collector is read-only and calls only Vast.ai Search Offers. It
does not create, rent, bid on, start, or destroy instances. `VAST_API_KEY` is
loaded from the explicit process environment first, then from the repository
root `.env` without overriding an existing value. This works independently of
whether the current working directory is the repository root or `src/`.

Each provider/pricing snapshot has one of these statuses:

- `SUCCESS`: the request and schema were valid, at least one offer was usable,
  the complete result was collected, and there were no warnings.
- `SUCCESS_WITH_WARNINGS`: the request produced at least one usable, complete
  offer set but had non-fatal record or GPU-model warnings.
- `API_KEY_MISSING`: no key was configured; no HTTP request was sent.
- `PROVIDER_ERROR`: the provider request failed, returned invalid JSON, or did
  not yield a complete result after bounded retries.
- `NO_MARKET_DATA`: the request and schema were valid and the complete filtered
  response contained zero offers. This is the only real zero-inventory status.
- `SCHEMA_ERROR`: the HTTP response arrived but its top-level structure was not
  a reliable offers response, or none of its source offers could be normalized.

The two generated layers have deliberately different grains:

- Raw offer history in `data/market_data/gpu_cloud_market_history.csv` retains
  every minute-level offer snapshot. Request outcomes, including failures, are
  retained separately in `gpu_cloud_market_fetch_log.csv` for audit and
  troubleshooting. Existing historical statuses are not rewritten.
- Daily signals in `data/signals/gpu_cloud_market_signals.csv` select the
  maximum eligible timestamp per date and pricing type. `SUCCESS` and
  `SUCCESS_WITH_WARNINGS` are eligible; a legacy `PARTIAL` is eligible only
  when `request_count > 0`, `offer_count > 0`, and `results_truncated=False`.
  A validated `NO_MARKET_DATA` snapshot is eligible only as a real zero.

On-demand and interruptible timestamps are selected independently and written
to the signal output. `source_snapshot_timestamp_utc` uses the selected
on-demand timestamp as the primary daily source, falling back to the selected
interruptible timestamp only when no eligible on-demand snapshot exists. If
interruptible collection fails while on-demand
succeeds, on-demand metrics remain valid, interruptible metrics and discount
remain empty, and the day is marked `PARTIAL_DAY`. `API_KEY_MISSING`,
`PROVIDER_ERROR`, and `SCHEMA_ERROR` never become zero inventory and never
contribute prices, counts, availability, or trend references. Seven- and
30-calendar-day trends use only the selected daily on-demand snapshots, with
the exact target date preferred and the configured plus/minus two-day
tolerance used when that date is absent. Price trends use the on-demand median
price per GPU-hour only. Offer-count trends are calculated independently per
tracked GPU model; their reference denominator must be greater than zero.
Missing references remain blank and are displayed as `Insufficient history`,
not zero percent.

The GPU update and signal build run near the start of the daily pipeline,
before the remaining market and signal steps. A GPU warning does not stop the
pipeline. A GPU API, provider, or schema failure also does not stop the other
steps or the final email; the GPU section says that data is unavailable and
never turns the failed request into zero offers or zero GPUs.

The unified daily email reads
`data/signals/gpu_cloud_market_signals.csv`, never the minute-level raw offer
history. For every tracked model it shows the 7- and 30-calendar-day on-demand
rental-price trends, visible GPU count, 7- and 30-calendar-day visible-offer
trends, and the stored supply signal. Visible GPUs and offers cover only the
filtered Vast.ai public marketplace and are not global capacity.

The supply signal is a conservative Vast.ai marginal public marketplace
indicator, not a measure of the whole GPU cloud market. It is written in the
signal layer using these rules, in priority order:

- `DATA_UNAVAILABLE`: the API or formal daily data is unavailable.
- `INSUFFICIENT_HISTORY`: either 30-day price or 30-day offer trend is missing.
- `OVERSUPPLY_WARNING`: 30-day price is at most -10%, 30-day offers are at
  least +20%, and either 7-day price is negative or 7-day offers are positive.
- `STABLE`: absolute 30-day price change is below 5% and absolute 30-day offer
  change is below 10%.
- `LOOSENING`: 30-day price is negative and 30-day offers are positive.
- `TIGHTENING`: 30-day price is positive and 30-day offers are negative.
- `MIXED`: every other available combination.

`INSUFFICIENT_HISTORY` is the expected initial condition while natural-day
history accumulates. The daily email no longer displays a `Generated Files`
section in either plain text or HTML; all configured files are still generated
and saved normally.

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
5. Daily 13-ETF leadership Top 3/Bottom 3 ranking render
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
`EMAIL_TO` settings to send one consolidated daily email after all data steps.
The VIX market-sentiment section, 30-, 90-, and 250-trading-day Top 3 and
Bottom 3 tables, and GPU supply section are rendered into that one message. The ranking step
does not send a second message. Its header describes an ETF leadership universe
rather than incorrectly calling all 13 entries primary sectors.

Standalone `--send-email` and `--send-email --force-email` ranking runs retain
their existing idempotent email-log behavior. The production ranking email log
is not changed when the ranking builder is used only to render the unified
daily email.

Email status is stored at:

```text
data/signals/sector_etf_ranking_email_log.csv
```

Returns in both the metrics and ranking CSVs remain decimal values. Email
templates render them as percentages and state that the rankings are a
quantitative summary, not an investment recommendation.

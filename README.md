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

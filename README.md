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

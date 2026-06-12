# Project Status

Current structure has been grouped by domain:

- Market data ingestion and history are under `src/market_data/`.
- Signal generation is under `src/signals/`.
- Earnings and IR event checks are under `src/events/`.
- Polymarket prediction market workflows are under `src/prediction_markets/`.
- Daily orchestration starts from `src/pipelines/run_daily_pipeline.py`.

Data files follow the same domain split under `data/`.

## Knowledge Layer

### Optical Test and Metrology

- [x] Add Optical Test and Metrology industry framework
- [x] Separate Optical Communication Test from Semiconductor Metrology
- [x] Add KEYS company framework
- [x] Add VIAV company framework
- [x] Add AEHR company framework
- [x] Add FORM company framework
- [ ] Verify company-specific financial exposure
- [ ] Convert verified relationships into machine-readable data
- [ ] Add Optical Test companies to company_master.csv
- [ ] Add sector-strength classification

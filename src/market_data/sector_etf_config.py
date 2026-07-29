"""
Validated configuration for the 11 GICS sector ETFs.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_FILE = BASE_DIR / "config" / "sector_etfs.yaml"
EXPECTED_ETF_COUNT = 11
FUND_HISTORY_FILENAME_PATTERN = re.compile(r"^[a-z0-9_]+\.csv$")


@dataclass(frozen=True)
class SectorETF:
    sector_id: str
    sector_name: str
    sector_name_cn: str
    ticker: str
    fund_history_filename: str


@dataclass(frozen=True)
class PriceHistorySettings:
    interval: str
    auto_adjust: bool
    initial_period: str
    incremental_overlap_days: int


@dataclass(frozen=True)
class SectorETFConfig:
    schema_version: int
    source_provider: str
    source_client: str
    state_street_nav_history_url_template: str
    price_history: PriceHistorySettings
    etfs: tuple[SectorETF, ...]


def _required_text(
    mapping: Mapping[str, Any],
    key: str,
    location: str,
) -> str:
    value = mapping.get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"{location}.{key} is required and cannot be empty")
    return str(value).strip()


def validate_fund_history_filename(
    filename: str,
    *,
    location: str = "fund_history_filename",
) -> str:
    """
    Validate a configured per-ETF CSV basename.

    The configured value is deliberately restricted to a lowercase basename,
    so joining it to the fund-history directory cannot escape that directory.
    """

    value = str(filename).strip()
    if (
        not value
        or Path(value).is_absolute()
        or Path(value).name != value
        or "/" in value
        or "\\" in value
        or ".." in value
        or not FUND_HISTORY_FILENAME_PATTERN.fullmatch(value)
    ):
        raise ValueError(
            f"{location} must be a safe lowercase .csv basename containing "
            "only letters, digits, and underscores"
        )
    return value


def load_sector_etf_config(
    config_path: Path | str = DEFAULT_CONFIG_FILE,
) -> SectorETFConfig:
    """
    Load the single source of truth for ETF and provider configuration.
    """

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Sector ETF config not found: {path}")

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid sector ETF YAML: {error}") from error

    if not isinstance(payload, dict):
        raise ValueError("Sector ETF config must contain a YAML mapping")

    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise ValueError("Sector ETF config schema_version must be 1")

    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError("Sector ETF config source must be a mapping")
    provider = _required_text(source, "provider", "source")
    client = _required_text(source, "client", "source")
    if provider != "yahoo_finance" or client != "yfinance":
        raise ValueError(
            "Sector ETF price source must be yahoo_finance with yfinance"
        )

    state_street = payload.get("state_street")
    if not isinstance(state_street, dict):
        raise ValueError("Sector ETF config state_street must be a mapping")
    nav_history_url_template = _required_text(
        state_street,
        "nav_history_url_template",
        "state_street",
    )
    try:
        rendered_url = nav_history_url_template.format(ticker_lower="xlk")
    except (KeyError, ValueError) as error:
        raise ValueError(
            "state_street.nav_history_url_template must contain a valid "
            "{ticker_lower} placeholder"
        ) from error
    if "{ticker_lower}" not in nav_history_url_template:
        raise ValueError(
            "state_street.nav_history_url_template must contain "
            "{ticker_lower}"
        )
    if not rendered_url.lower().endswith("xlk.xlsx"):
        raise ValueError(
            "state_street.nav_history_url_template must render an XLSX URL"
        )

    price_payload = payload.get("price_history")
    if not isinstance(price_payload, dict):
        raise ValueError("Sector ETF price_history must be a mapping")
    interval = _required_text(price_payload, "interval", "price_history")
    initial_period = _required_text(
        price_payload,
        "initial_period",
        "price_history",
    )
    auto_adjust = price_payload.get("auto_adjust")
    if auto_adjust is not False:
        raise ValueError(
            "price_history.auto_adjust must be false so close and adj_close "
            "remain distinct"
        )
    overlap_days = price_payload.get("incremental_overlap_days", 5)
    if (
        isinstance(overlap_days, bool)
        or not isinstance(overlap_days, int)
        or overlap_days < 0
    ):
        raise ValueError(
            "price_history.incremental_overlap_days must be a non-negative "
            "integer"
        )

    raw_etfs = payload.get("etfs")
    if not isinstance(raw_etfs, list):
        raise ValueError("Sector ETF config etfs must be a list")

    etfs: list[SectorETF] = []
    for index, raw_etf in enumerate(raw_etfs):
        location = f"etfs[{index}]"
        if not isinstance(raw_etf, dict):
            raise ValueError(f"{location} must be a mapping")
        etfs.append(
            SectorETF(
                sector_id=_required_text(raw_etf, "sector_id", location),
                sector_name=_required_text(raw_etf, "sector_name", location),
                sector_name_cn=_required_text(
                    raw_etf,
                    "sector_name_cn",
                    location,
                ),
                ticker=_required_text(raw_etf, "ticker", location).upper(),
                fund_history_filename=validate_fund_history_filename(
                    _required_text(
                        raw_etf,
                        "fund_history_filename",
                        location,
                    ),
                    location=f"{location}.fund_history_filename",
                ),
            )
        )

    if len(etfs) != EXPECTED_ETF_COUNT:
        raise ValueError(
            f"Sector ETF config must contain exactly {EXPECTED_ETF_COUNT} ETFs; "
            f"found {len(etfs)}"
        )

    sector_ids = [etf.sector_id for etf in etfs]
    tickers = [etf.ticker for etf in etfs]
    fund_history_filenames = [
        etf.fund_history_filename
        for etf in etfs
    ]
    if len(set(sector_ids)) != len(sector_ids):
        raise ValueError("Sector ETF config contains duplicate sector_id values")
    if len(set(tickers)) != len(tickers):
        raise ValueError("Sector ETF config contains duplicate ticker values")
    if len(set(fund_history_filenames)) != len(fund_history_filenames):
        raise ValueError(
            "Sector ETF config contains duplicate fund_history_filename values"
        )

    return SectorETFConfig(
        schema_version=schema_version,
        source_provider=provider,
        source_client=client,
        state_street_nav_history_url_template=nav_history_url_template,
        price_history=PriceHistorySettings(
            interval=interval,
            auto_adjust=auto_adjust,
            initial_period=initial_period,
            incremental_overlap_days=overlap_days,
        ),
        etfs=tuple(etfs),
    )

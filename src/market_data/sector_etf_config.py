"""Validated provider and universe configuration for sector leadership ETFs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_FILE = BASE_DIR / "config" / "sector_etfs.yaml"
EXPECTED_PRIMARY_SECTOR_COUNT = 11
EXPECTED_LEADERSHIP_COUNT = 13
SUPPORTED_FUND_DATA_PROVIDERS = {"state_street", "ishares"}
SUPPORTED_CLASSIFICATION_LEVELS = {"sector", "industry"}
FUND_HISTORY_FILENAME_PATTERN = re.compile(r"^[a-z0-9_]+\.csv$")


@dataclass(frozen=True)
class SectorETF:
    sector_id: str
    sector_name: str
    sector_name_cn: str
    ticker: str
    classification_level: str
    parent_sector_id: str | None
    fund_data_provider: str
    ishares_product_id: int | None
    product_page_slug: str | None
    fund_history_filename: str
    metrics_filename: str


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
    ishares_product_page_url_template: str
    ishares_fund_download_url_template: str
    price_history: PriceHistorySettings
    primary_sector_tickers: tuple[str, ...]
    leadership_tickers: tuple[str, ...]
    etfs: tuple[SectorETF, ...]

    def _etfs_for_tickers(
        self,
        tickers: tuple[str, ...],
    ) -> tuple[SectorETF, ...]:
        by_ticker = {etf.ticker: etf for etf in self.etfs}
        return tuple(by_ticker[ticker] for ticker in tickers)

    @property
    def primary_sector_etfs(self) -> tuple[SectorETF, ...]:
        return self._etfs_for_tickers(self.primary_sector_tickers)

    @property
    def leadership_etfs(self) -> tuple[SectorETF, ...]:
        return self._etfs_for_tickers(self.leadership_tickers)

    @property
    def leadership_overlay_etfs(self) -> tuple[SectorETF, ...]:
        primary = set(self.primary_sector_tickers)
        return tuple(
            etf for etf in self.leadership_etfs if etf.ticker not in primary
        )

    @property
    def state_street_etfs(self) -> tuple[SectorETF, ...]:
        return tuple(
            etf
            for etf in self.etfs
            if etf.fund_data_provider == "state_street"
        )

    @property
    def ishares_etfs(self) -> tuple[SectorETF, ...]:
        return tuple(
            etf for etf in self.etfs if etf.fund_data_provider == "ishares"
        )


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
    if schema_version != 2:
        raise ValueError("Sector ETF config schema_version must be 2")

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

    ishares = payload.get("ishares")
    if not isinstance(ishares, dict):
        raise ValueError("Sector ETF config ishares must be a mapping")
    ishares_product_page_url_template = _required_text(
        ishares,
        "product_page_url_template",
        "ishares",
    )
    ishares_fund_download_url_template = _required_text(
        ishares,
        "fund_download_url_template",
        "ishares",
    )
    for name, template in (
        ("product_page_url_template", ishares_product_page_url_template),
        ("fund_download_url_template", ishares_fund_download_url_template),
    ):
        for placeholder in ("{product_id}",):
            if placeholder not in template:
                raise ValueError(
                    f"ishares.{name} must contain {placeholder}"
                )
    if "{product_slug}" not in ishares_product_page_url_template:
        raise ValueError(
            "ishares.product_page_url_template must contain {product_slug}"
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
        classification_level = _required_text(
            raw_etf,
            "classification_level",
            location,
        ).lower()
        if classification_level not in SUPPORTED_CLASSIFICATION_LEVELS:
            raise ValueError(
                f"{location}.classification_level must be one of "
                f"{sorted(SUPPORTED_CLASSIFICATION_LEVELS)}"
            )
        fund_data_provider = _required_text(
            raw_etf,
            "fund_data_provider",
            location,
        ).lower()
        if fund_data_provider not in SUPPORTED_FUND_DATA_PROVIDERS:
            raise ValueError(
                f"{location}.fund_data_provider must be one of "
                f"{sorted(SUPPORTED_FUND_DATA_PROVIDERS)}"
            )
        parent_sector_id_raw = raw_etf.get("parent_sector_id")
        parent_sector_id = (
            str(parent_sector_id_raw).strip()
            if parent_sector_id_raw is not None
            else None
        )
        if parent_sector_id == "":
            parent_sector_id = None
        product_slug_raw = raw_etf.get("product_page_slug")
        product_page_slug = (
            str(product_slug_raw).strip()
            if product_slug_raw is not None
            else None
        )
        product_id_raw = raw_etf.get("ishares_product_id")
        product_id = (
            product_id_raw
            if isinstance(product_id_raw, int)
            and not isinstance(product_id_raw, bool)
            and product_id_raw > 0
            else None
        )
        if fund_data_provider == "ishares":
            if product_id is None or not product_page_slug:
                raise ValueError(
                    f"{location} iShares ETF requires a positive "
                    "ishares_product_id and product_page_slug"
                )
        elif product_id_raw is not None or product_page_slug:
            raise ValueError(
                f"{location} State Street ETF cannot define iShares metadata"
            )
        if classification_level == "industry" and not parent_sector_id:
            raise ValueError(
                f"{location} industry ETF requires parent_sector_id"
            )
        if classification_level == "sector" and parent_sector_id:
            raise ValueError(
                f"{location} sector ETF cannot define parent_sector_id"
            )

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
                classification_level=classification_level,
                parent_sector_id=parent_sector_id,
                fund_data_provider=fund_data_provider,
                ishares_product_id=product_id,
                product_page_slug=product_page_slug,
                fund_history_filename=validate_fund_history_filename(
                    _required_text(
                        raw_etf,
                        "fund_history_filename",
                        location,
                    ),
                    location=f"{location}.fund_history_filename",
                ),
                metrics_filename=validate_fund_history_filename(
                    _required_text(raw_etf, "metrics_filename", location),
                    location=f"{location}.metrics_filename",
                ),
            )
        )

    if len(etfs) != EXPECTED_LEADERSHIP_COUNT:
        raise ValueError(
            "Sector ETF config must contain exactly "
            f"{EXPECTED_LEADERSHIP_COUNT} ETFs; "
            f"found {len(etfs)}"
        )

    universes = payload.get("universes")
    if not isinstance(universes, dict):
        raise ValueError("Sector ETF config universes must be a mapping")

    def load_universe(name: str, expected_count: int) -> tuple[str, ...]:
        raw_values = universes.get(name)
        if not isinstance(raw_values, list):
            raise ValueError(f"universes.{name} must be a list")
        values = tuple(str(value).strip().upper() for value in raw_values)
        if any(not value for value in values):
            raise ValueError(f"universes.{name} contains an empty ticker")
        if len(values) != expected_count:
            raise ValueError(
                f"universes.{name} must contain exactly {expected_count} "
                f"tickers; found {len(values)}"
            )
        if len(set(values)) != len(values):
            raise ValueError(f"universes.{name} contains duplicate tickers")
        return values

    primary_sector_tickers = load_universe(
        "primary_sector",
        EXPECTED_PRIMARY_SECTOR_COUNT,
    )
    leadership_tickers = load_universe(
        "leadership",
        EXPECTED_LEADERSHIP_COUNT,
    )

    sector_ids = [etf.sector_id for etf in etfs]
    tickers = [etf.ticker for etf in etfs]
    fund_history_filenames = [
        etf.fund_history_filename
        for etf in etfs
    ]
    metrics_filenames = [etf.metrics_filename for etf in etfs]
    if len(set(sector_ids)) != len(sector_ids):
        raise ValueError("Sector ETF config contains duplicate sector_id values")
    if len(set(tickers)) != len(tickers):
        raise ValueError("Sector ETF config contains duplicate ticker values")
    if len(set(fund_history_filenames)) != len(fund_history_filenames):
        raise ValueError(
            "Sector ETF config contains duplicate fund_history_filename values"
        )
    if len(set(metrics_filenames)) != len(metrics_filenames):
        raise ValueError(
            "Sector ETF config contains duplicate metrics_filename values"
        )

    configured_tickers = set(tickers)
    for name, universe_tickers in (
        ("primary_sector", primary_sector_tickers),
        ("leadership", leadership_tickers),
    ):
        unknown = sorted(set(universe_tickers) - configured_tickers)
        if unknown:
            raise ValueError(
                f"universes.{name} contains unconfigured ticker(s): "
                + ", ".join(unknown)
            )
    if not set(primary_sector_tickers).issubset(leadership_tickers):
        raise ValueError(
            "universes.primary_sector must be a subset of universes.leadership"
        )
    if set(leadership_tickers) != configured_tickers:
        raise ValueError(
            "universes.leadership must contain every configured ETF exactly once"
        )

    by_ticker = {etf.ticker: etf for etf in etfs}
    invalid_primary = [
        ticker
        for ticker in primary_sector_tickers
        if by_ticker[ticker].classification_level != "sector"
        or by_ticker[ticker].fund_data_provider != "state_street"
    ]
    if invalid_primary:
        raise ValueError(
            "Primary-sector universe must contain only State Street sector "
            "ETFs: " + ", ".join(invalid_primary)
        )
    invalid_overlays = [
        etf.ticker
        for etf in (
            by_ticker[ticker]
            for ticker in set(leadership_tickers)
            - set(primary_sector_tickers)
        )
        if etf.classification_level != "industry"
        or etf.fund_data_provider != "ishares"
    ]
    if invalid_overlays:
        raise ValueError(
            "Leadership overlays must be iShares industry ETFs: "
            + ", ".join(sorted(invalid_overlays))
        )
    if {
        etf.ticker
        for etf in etfs
        if etf.fund_data_provider == "state_street"
    } != set(primary_sector_tickers):
        raise ValueError(
            "State Street ETFs must exactly match universes.primary_sector"
        )
    if {
        etf.ticker
        for etf in etfs
        if etf.fund_data_provider == "ishares"
    } != set(leadership_tickers) - set(primary_sector_tickers):
        raise ValueError(
            "iShares ETFs must exactly match the leadership overlay universe"
        )
    product_ids = [
        etf.ishares_product_id
        for etf in etfs
        if etf.ishares_product_id is not None
    ]
    if len(set(product_ids)) != len(product_ids):
        raise ValueError("Sector ETF config contains duplicate iShares product IDs")

    return SectorETFConfig(
        schema_version=schema_version,
        source_provider=provider,
        source_client=client,
        state_street_nav_history_url_template=nav_history_url_template,
        ishares_product_page_url_template=ishares_product_page_url_template,
        ishares_fund_download_url_template=ishares_fund_download_url_template,
        price_history=PriceHistorySettings(
            interval=interval,
            auto_adjust=auto_adjust,
            initial_period=initial_period,
            incremental_overlap_days=overlap_days,
        ),
        primary_sector_tickers=primary_sector_tickers,
        leadership_tickers=leadership_tickers,
        etfs=tuple(etfs),
    )

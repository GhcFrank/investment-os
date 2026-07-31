"""Build the formal current VIX market-sentiment signal."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pandas as pd

from market_data.update_sentiment_indicators import (
    VIX_COLUMNS,
    VIX_CURRENT_FILE,
    numeric_value,
)
from market_data.vix_sentiment_config import (
    VIX_DATA_STATUSES,
    VIX_MARKET_SENTIMENT_SIGNAL_COLUMNS,
    VIX_MARKET_SENTIMENT_SIGNAL_FILE,
)
from utils.csv_utils import atomic_write_csv
from utils.date_utils import today_et_str


@dataclass(frozen=True)
class VIXMarketSentimentSignalsSummary:
    output_file: Path
    data_date: str
    data_status: str
    file_written: bool

    def format(self) -> str:
        return "\n".join(
            [
                "VIX market sentiment signal summary:",
                f"- data date: {self.data_date}",
                f"- data status: {self.data_status}",
                f"- output written: {self.file_written}",
            ]
        )


def _read_current_vix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"VIX current file not found: {path}")
    try:
        rows = pd.read_csv(path, dtype=str)
    except pd.errors.EmptyDataError as error:
        raise ValueError("VIX current file is empty") from error
    if rows.empty or list(rows.columns) != VIX_COLUMNS:
        raise ValueError("VIX current file has an invalid schema")
    return rows.tail(1).copy()


def _regime_and_signal(level: object) -> tuple[str, str]:
    normalized = str(level or "").strip().lower()
    mapping = {
        "calm": ("CALM", "LOW_VOLATILITY"),
        "normal": ("NORMAL", "NEUTRAL"),
        "risk_off": ("ELEVATED", "RISK_OFF_PRESSURE"),
        "stress": ("STRESS", "HIGH_STRESS"),
    }
    return mapping.get(normalized, ("UNKNOWN", "NEUTRAL"))


def build_vix_market_sentiment_signal(
    current_vix: pd.DataFrame,
    *,
    current_et_date: str | None = None,
) -> pd.DataFrame:
    """Map the existing VIX observation and point changes into one signal."""

    if current_vix.empty:
        raise ValueError("VIX current data cannot be empty")
    row = current_vix.tail(1).iloc[0]
    data_date = str(row.get("date", "")).strip()
    vix = numeric_value(row.get("vix"))
    source_status = str(row.get("status", "")).strip().lower()
    changes = {
        column: numeric_value(row.get(column))
        for column in ("change_1d", "change_5d", "change_20d")
    }
    today = current_et_date or today_et_str()
    stale = bool(data_date and data_date != today)
    notes: list[str] = []

    if source_status != "ok" or not data_date or vix is None:
        data_status = "DATA_UNAVAILABLE"
        notes.append("formal VIX observation unavailable")
    else:
        missing = [name for name, value in changes.items() if value is None]
        if missing:
            data_status = "INSUFFICIENT_HISTORY"
            notes.append("missing VIX point change(s): " + ", ".join(missing))
        elif stale:
            data_status = "SUCCESS_WITH_WARNINGS"
            notes.append("latest VIX observation is not dated today ET")
        else:
            data_status = "SUCCESS"
    if data_status not in VIX_DATA_STATUSES:
        raise AssertionError(f"Unsupported VIX data status: {data_status}")

    regime, signal = _regime_and_signal(row.get("level"))
    return pd.DataFrame(
        [
            {
                "date": data_date,
                "vix": vix,
                "change_1d": changes["change_1d"],
                "change_5d": changes["change_5d"],
                "change_20d": changes["change_20d"],
                "sentiment_regime": regime,
                "signal": signal,
                "source": str(row.get("source", "")).strip(),
                "source_status": source_status,
                "data_status": data_status,
                "stale": stale,
                "updated_at": str(row.get("updated_at", "")).strip(),
                "data_quality_notes": "; ".join(notes),
            }
        ],
        columns=VIX_MARKET_SENTIMENT_SIGNAL_COLUMNS,
    )


def _canonical_csv(signals: pd.DataFrame) -> str:
    if list(signals.columns) != VIX_MARKET_SENTIMENT_SIGNAL_COLUMNS:
        raise ValueError("VIX sentiment signal schema is not stable")
    buffer = StringIO()
    signals.to_csv(
        buffer,
        columns=VIX_MARKET_SENTIMENT_SIGNAL_COLUMNS,
        index=False,
        float_format="%.15g",
        na_rep="",
        lineterminator="\n",
    )
    return buffer.getvalue()


def write_vix_market_sentiment_signal(
    signals: pd.DataFrame,
    output_file: Path | str,
) -> bool:
    path = Path(output_file)
    content = _canonical_csv(signals)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False

    def validate_temp_file(temp_path: Path) -> None:
        if temp_path.read_text(encoding="utf-8") != content:
            raise ValueError("Temporary VIX signal verification failed")

    atomic_write_csv(
        signals,
        path,
        VIX_MARKET_SENTIMENT_SIGNAL_COLUMNS,
        float_format="%.15g",
        na_rep="",
        lineterminator="\n",
        validate_temp_file=validate_temp_file,
    )
    return True


def run_vix_market_sentiment_signals_update(
    *,
    current_file: Path | str = VIX_CURRENT_FILE,
    output_file: Path | str = VIX_MARKET_SENTIMENT_SIGNAL_FILE,
    current_et_date: str | None = None,
) -> VIXMarketSentimentSignalsSummary:
    current = _read_current_vix(Path(current_file))
    signals = build_vix_market_sentiment_signal(
        current,
        current_et_date=current_et_date,
    )
    output_path = Path(output_file)
    written = write_vix_market_sentiment_signal(signals, output_path)
    row = signals.iloc[-1]
    return VIXMarketSentimentSignalsSummary(
        output_file=output_path,
        data_date=str(row["date"]),
        data_status=str(row["data_status"]),
        file_written=written,
    )


def main() -> None:
    print(run_vix_market_sentiment_signals_update().format())


if __name__ == "__main__":
    main()

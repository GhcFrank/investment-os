"""
Shared helpers for maintaining daily CSV history files.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd


def upsert_daily_history(
    daily_df: pd.DataFrame,
    history_file: Path,
) -> None:
    """
    Insert or replace today's rows in a daily history CSV.

    A history dataset must contain either a ticker or subtheme column. Running
    the same pipeline more than once per day replaces that entity's earlier
    row instead of creating duplicates.
    """

    key_column = next(
        (
            column
            for column in ("ticker", "subtheme")
            if column in daily_df.columns
        ),
        None,
    )

    if key_column is None:
        raise ValueError(
            "Daily history data must contain a 'ticker' or 'subtheme' column"
        )

    history_df = daily_df.copy()
    history_df.insert(
        0,
        "date",
        datetime.now().strftime("%Y-%m-%d"),
    )

    if history_file.exists():
        existing = pd.read_csv(history_file)
        combined = pd.concat(
            [existing, history_df],
            ignore_index=True,
        )
    else:
        history_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        combined = history_df

    combined = combined.drop_duplicates(
        subset=["date", key_column],
        keep="last",
    )

    columns = [
        "date",
        *(
            column
            for column in combined.columns
            if column != "date"
        ),
    ]

    combined.to_csv(
        history_file,
        columns=columns,
        index=False,
    )

    print(f"\nUpdated history file:\n{history_file}\n")

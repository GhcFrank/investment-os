"""
Shared helpers for writing stable CSV files safely.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd


def atomic_write_csv(
    df: pd.DataFrame,
    path: Path,
    columns: list[str],
) -> None:
    """
    Write a CSV with a stable schema via a same-directory atomic replace.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    output = df.copy()

    for column in columns:
        if column not in output.columns:
            output[column] = ""

    output = output.reindex(columns=columns)
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
        encoding="utf-8",
    )
    temp_path = Path(temp_file.name)
    temp_file.close()

    try:
        output.to_csv(temp_path, index=False, encoding="utf-8")
        target_mode = (
            path.stat().st_mode & 0o777
            if path.exists()
            else 0o644
        )
        os.chmod(temp_path, target_mode)
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

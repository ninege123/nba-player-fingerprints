"""Schema helpers for raw and engineered basketball data."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def missing_columns(frame: pd.DataFrame, required_columns: Sequence[str]) -> list[str]:
    """Return required columns that are absent from a DataFrame."""
    return [column for column in required_columns if column not in frame.columns]


def validate_columns(
    frame: pd.DataFrame,
    required_columns: Sequence[str],
    dataset_name: str = "dataset",
) -> None:
    """Raise a clear error if a DataFrame is missing required columns."""
    missing = missing_columns(frame, required_columns)
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")

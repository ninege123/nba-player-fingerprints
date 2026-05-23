"""Normalization helpers for player fingerprint features."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def min_max_scale_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Return a copy of `frame` with selected feature columns min-max scaled.

    Constant columns are set to 0.0 because they do not help distinguish player
    profiles within the comparison set.
    """
    missing_columns = [column for column in feature_columns if column not in frame.columns]
    if missing_columns:
        raise KeyError(f"Missing feature columns: {missing_columns}")

    scaled = frame.copy()
    for column in feature_columns:
        values = pd.to_numeric(scaled[column], errors="coerce")
        min_value = values.min(skipna=True)
        max_value = values.max(skipna=True)

        if pd.isna(min_value) or pd.isna(max_value) or np.isclose(max_value, min_value):
            scaled[column] = 0.0
            continue

        scaled[column] = (values - min_value) / (max_value - min_value)

    return scaled


def normalize_vector(vector: Sequence[float]) -> np.ndarray:
    """Return an L2-normalized numpy vector.

    A zero vector remains zero so downstream cosine calculations can handle
    inactive or empty profiles explicitly.
    """
    array = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(array)
    if np.isclose(norm, 0.0):
        return np.zeros_like(array, dtype=float)
    return array / norm

"""Similarity scoring for player fingerprints."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from nba_fingerprints.features.normalization import normalize_vector


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute cosine similarity between two fingerprint vectors."""
    left_vector = normalize_vector(left)
    right_vector = normalize_vector(right)

    if np.isclose(np.linalg.norm(left_vector), 0.0) or np.isclose(np.linalg.norm(right_vector), 0.0):
        return 0.0

    return float(np.dot(left_vector, right_vector))

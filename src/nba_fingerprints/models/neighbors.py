"""Nearest-neighbor search for player fingerprints."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from nba_fingerprints.features.normalization import normalize_vector
from nba_fingerprints.features.player_season import FINGERPRINT_FEATURE_COLUMNS
from nba_fingerprints.features.schema import validate_columns


DEFAULT_PLAYER_ID_COLUMNS = [
    "season",
    "player_id",
    "player_name",
    "team_abbreviation",
]


def player_similarity_matrix(
    fingerprints: pd.DataFrame,
    feature_columns: Sequence[str] = FINGERPRINT_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Compute a square cosine-similarity matrix for a fingerprint table."""
    validate_columns(fingerprints, feature_columns, dataset_name="fingerprints")

    vectors = fingerprints.loc[:, feature_columns].to_numpy(dtype=float)
    normalized = np.vstack([normalize_vector(row) for row in vectors])
    similarities = normalized @ normalized.T

    labels = fingerprints["player_name"].tolist() if "player_name" in fingerprints.columns else fingerprints.index.tolist()
    return pd.DataFrame(similarities, index=labels, columns=labels)


def find_nearest_neighbors(
    fingerprints: pd.DataFrame,
    top_n: int = 5,
    feature_columns: Sequence[str] = FINGERPRINT_FEATURE_COLUMNS,
    id_columns: Sequence[str] = DEFAULT_PLAYER_ID_COLUMNS,
) -> pd.DataFrame:
    """Find each player's top-N most similar player-season fingerprints."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    validate_columns(fingerprints, [*id_columns, *feature_columns], dataset_name="fingerprints")

    vectors = fingerprints.loc[:, feature_columns].to_numpy(dtype=float)
    normalized = np.vstack([normalize_vector(row) for row in vectors])
    similarities = normalized @ normalized.T

    rows: list[dict[str, object]] = []
    for player_index, player in fingerprints.reset_index(drop=True).iterrows():
        order = np.argsort(similarities[player_index])[::-1]
        neighbor_indexes = [index for index in order if index != player_index][:top_n]

        for rank, neighbor_index in enumerate(neighbor_indexes, start=1):
            neighbor = fingerprints.iloc[neighbor_index]
            rows.append(
                {
                    "season": player["season"],
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                    "team_abbreviation": player["team_abbreviation"],
                    "neighbor_rank": rank,
                    "neighbor_season": neighbor["season"],
                    "neighbor_player_id": neighbor["player_id"],
                    "neighbor_player_name": neighbor["player_name"],
                    "neighbor_team_abbreviation": neighbor["team_abbreviation"],
                    "cosine_similarity": float(similarities[player_index, neighbor_index]),
                }
            )

    return pd.DataFrame(rows)

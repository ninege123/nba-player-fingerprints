"""Build normalized player fingerprint tables from engineered features."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from nba_fingerprints.features.normalization import min_max_scale_features
from nba_fingerprints.features.player_season import FINGERPRINT_FEATURE_COLUMNS
from nba_fingerprints.features.schema import validate_columns


DEFAULT_ID_COLUMNS = [
    "season",
    "player_id",
    "player_name",
    "team_abbreviation",
]


def build_fingerprint_table(
    player_features: pd.DataFrame,
    feature_columns: Sequence[str] = FINGERPRINT_FEATURE_COLUMNS,
    id_columns: Sequence[str] = DEFAULT_ID_COLUMNS,
) -> pd.DataFrame:
    """Return player metadata plus normalized fingerprint feature columns."""
    validate_columns(player_features, [*id_columns, *feature_columns], dataset_name="player features")

    scaled = min_max_scale_features(player_features, feature_columns)
    output_columns = [*id_columns, *feature_columns]
    return scaled.loc[:, output_columns].reset_index(drop=True)

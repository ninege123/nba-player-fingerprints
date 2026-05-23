"""Position labels and reference fingerprints."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from nba_fingerprints.features.player_season import FINGERPRINT_FEATURE_COLUMNS
from nba_fingerprints.features.schema import validate_columns
from nba_fingerprints.models.similarity import cosine_similarity


POSITION_LABELS = ["G", "F", "C"]


def attach_position_labels(player_features: pd.DataFrame, player_index: pd.DataFrame) -> pd.DataFrame:
    """Attach listed NBA position labels to player-season features."""
    validate_columns(player_features, ["player_id", "team_id"], dataset_name="player features")
    validate_columns(player_index, ["PERSON_ID", "TEAM_ID", "POSITION"], dataset_name="player index")

    positions = player_index.loc[:, ["PERSON_ID", "POSITION"]].dropna(subset=["PERSON_ID"]).rename(
        columns={
            "PERSON_ID": "player_id",
            "POSITION": "position",
        }
    )
    positions = positions.drop_duplicates(subset=["player_id"], keep="first")
    positions["primary_position"] = positions["position"].apply(primary_position)

    merged = player_features.merge(positions, on="player_id", how="left")
    return merged


def primary_position(position: object) -> str:
    """Return a normalized primary NBA-listed position from a position label."""
    if not isinstance(position, str) or not position.strip():
        return "UNK"

    first_position = position.replace(" ", "").split("-")[0].upper()
    return first_position if first_position in POSITION_LABELS else "UNK"


def build_position_references(
    fingerprints: pd.DataFrame,
    feature_columns: Sequence[str] = FINGERPRINT_FEATURE_COLUMNS,
    min_players: int = 1,
) -> pd.DataFrame:
    """Average player fingerprints into position reference fingerprints."""
    validate_columns(fingerprints, ["primary_position", *feature_columns], dataset_name="fingerprints")

    eligible = fingerprints[fingerprints["primary_position"].isin(POSITION_LABELS)].copy()
    references = eligible.groupby("primary_position", as_index=False)[list(feature_columns)].mean()
    counts = eligible.groupby("primary_position", as_index=False).size().rename(columns={"size": "player_count"})
    references = references.merge(counts, on="primary_position")
    references = references[references["player_count"] >= min_players].copy()
    return references.rename(columns={"primary_position": "reference_position"}).reset_index(drop=True)


def score_position_similarity(
    fingerprints: pd.DataFrame,
    position_references: pd.DataFrame,
    feature_columns: Sequence[str] = FINGERPRINT_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Score every player fingerprint against each position reference."""
    validate_columns(
        fingerprints,
        ["season", "player_id", "player_name", "team_abbreviation", "primary_position", *feature_columns],
        dataset_name="fingerprints",
    )
    validate_columns(position_references, ["reference_position", *feature_columns], dataset_name="position references")

    rows: list[dict[str, object]] = []
    for _, player in fingerprints.iterrows():
        player_vector = player.loc[list(feature_columns)].to_numpy(dtype=float)
        for _, reference in position_references.iterrows():
            reference_vector = reference.loc[list(feature_columns)].to_numpy(dtype=float)
            rows.append(
                {
                    "season": player["season"],
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                    "team_abbreviation": player["team_abbreviation"],
                    "listed_position": player["position"],
                    "primary_position": player["primary_position"],
                    "reference_position": reference["reference_position"],
                    "cosine_similarity": cosine_similarity(player_vector, reference_vector),
                }
            )

    scores = pd.DataFrame(rows)
    scores["position_rank"] = scores.groupby(["season", "player_id", "team_abbreviation"])["cosine_similarity"].rank(
        method="first",
        ascending=False,
    )
    return scores.sort_values(["player_name", "position_rank"]).reset_index(drop=True)

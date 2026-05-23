"""Manual basketball archetype fingerprints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from nba_fingerprints.features.player_season import FINGERPRINT_FEATURE_COLUMNS
from nba_fingerprints.features.schema import validate_columns
from nba_fingerprints.models.similarity import cosine_similarity


ARCHETYPE_WEIGHTS: dict[str, dict[str, float]] = {
    "floor_general": {
        "assist_pct": 1.0,
        "assist_to_turnover": 0.9,
        "assists_per_36": 0.8,
        "minutes_per_game": 0.5,
        "true_shooting_pct": 0.4,
        "usage_rate": 0.3,
    },
    "scoring_guard": {
        "points_per_36": 1.0,
        "usage_rate": 0.9,
        "three_point_attempt_rate": 0.8,
        "free_throw_attempt_rate": 0.5,
        "assists_per_36": 0.4,
        "true_shooting_pct": 0.4,
    },
    "three_and_d_wing": {
        "three_point_attempt_rate": 1.0,
        "steals_per_36": 0.8,
        "minutes_per_game": 0.6,
        "true_shooting_pct": 0.5,
        "personal_fouls_per_36": 0.3,
    },
    "point_forward": {
        "assist_pct": 0.9,
        "total_rebound_pct": 0.8,
        "defensive_rebound_pct": 0.6,
        "assists_per_36": 0.6,
        "points_per_36": 0.5,
        "minutes_per_game": 0.5,
    },
    "stretch_big": {
        "blocks_per_36": 0.7,
        "total_rebound_pct": 0.7,
        "three_point_attempt_rate": 0.7,
        "defensive_rebound_pct": 0.6,
        "true_shooting_pct": 0.5,
    },
    "rim_running_center": {
        "total_rebound_pct": 1.0,
        "offensive_rebound_pct": 0.9,
        "blocks_per_36": 0.8,
        "true_shooting_pct": 0.7,
        "player_impact_estimate": 0.6,
        "free_throw_attempt_rate": 0.5,
    },
    "interior_defensive_big": {
        "blocks_per_36": 1.0,
        "defensive_rebound_pct": 0.9,
        "total_rebound_pct": 0.8,
        "personal_fouls_per_36": 0.5,
        "player_impact_estimate": 0.5,
        "minutes_per_game": 0.4,
    },
    "high_usage_creator": {
        "usage_rate": 1.0,
        "points_per_36": 1.0,
        "assist_pct": 0.8,
        "turnovers_per_36": 0.7,
        "free_throw_attempt_rate": 0.6,
        "minutes_per_game": 0.6,
    },
}


def build_archetype_references(
    archetype_weights: Mapping[str, Mapping[str, float]] = ARCHETYPE_WEIGHTS,
    feature_columns: Sequence[str] = FINGERPRINT_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Convert manual archetype feature weights into a reference table."""
    rows: list[dict[str, object]] = []
    feature_set = set(feature_columns)
    for archetype_name, weights in archetype_weights.items():
        unknown_features = sorted(set(weights) - feature_set)
        if unknown_features:
            raise ValueError(f"{archetype_name} has unknown feature weights: {unknown_features}")

        row: dict[str, object] = {"archetype": archetype_name}
        row.update({feature: float(weights.get(feature, 0.0)) for feature in feature_columns})
        rows.append(row)

    return pd.DataFrame(rows)


def score_archetype_similarity(
    fingerprints: pd.DataFrame,
    archetype_references: pd.DataFrame,
    feature_columns: Sequence[str] = FINGERPRINT_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Score every player fingerprint against each manual archetype reference."""
    validate_columns(
        fingerprints,
        ["season", "player_id", "player_name", "team_abbreviation", *feature_columns],
        dataset_name="fingerprints",
    )
    validate_columns(archetype_references, ["archetype", *feature_columns], dataset_name="archetype references")

    rows: list[dict[str, object]] = []
    for _, player in fingerprints.iterrows():
        player_vector = player.loc[list(feature_columns)].to_numpy(dtype=float)
        for _, archetype in archetype_references.iterrows():
            archetype_vector = archetype.loc[list(feature_columns)].to_numpy(dtype=float)
            rows.append(
                {
                    "season": player["season"],
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                    "team_abbreviation": player["team_abbreviation"],
                    "archetype": archetype["archetype"],
                    "cosine_similarity": cosine_similarity(player_vector, archetype_vector),
                }
            )

    scores = pd.DataFrame(rows)
    scores["archetype_rank"] = scores.groupby(["season", "player_id", "team_abbreviation"])["cosine_similarity"].rank(
        method="first",
        ascending=False,
    )
    return scores.sort_values(["player_name", "archetype_rank"]).reset_index(drop=True)


def explain_top_archetype_matches(
    fingerprints: pd.DataFrame,
    archetype_references: pd.DataFrame,
    archetype_scores: pd.DataFrame,
    feature_columns: Sequence[str] = FINGERPRINT_FEATURE_COLUMNS,
    rank: int = 1,
    top_n_features: int = 3,
) -> pd.DataFrame:
    """Explain top player-archetype matches with supporting features and gaps."""
    validate_columns(
        fingerprints,
        ["season", "player_id", "team_abbreviation", *feature_columns],
        dataset_name="fingerprints",
    )
    validate_columns(archetype_references, ["archetype", *feature_columns], dataset_name="archetype references")
    validate_columns(
        archetype_scores,
        [
            "season",
            "player_id",
            "player_name",
            "team_abbreviation",
            "archetype",
            "cosine_similarity",
            "archetype_rank",
        ],
        dataset_name="archetype scores",
    )

    target_scores = archetype_scores[archetype_scores["archetype_rank"] == rank].copy()
    merged = target_scores.merge(
        fingerprints.loc[:, ["season", "player_id", "team_abbreviation", *feature_columns]],
        on=["season", "player_id", "team_abbreviation"],
        how="left",
    ).merge(
        archetype_references,
        on="archetype",
        how="left",
        suffixes=("_player", "_archetype"),
    )

    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        player_values = {feature: float(row[f"{feature}_player"]) for feature in feature_columns}
        archetype_values = {feature: float(row[f"{feature}_archetype"]) for feature in feature_columns}

        support = {
            feature: min(player_values[feature], archetype_values[feature])
            for feature in feature_columns
            if archetype_values[feature] > 0
        }
        gaps = {
            feature: abs(player_values[feature] - archetype_values[feature])
            for feature in feature_columns
            if archetype_values[feature] > 0
        }
        rows.append(
            {
                "season": row["season"],
                "player_id": row["player_id"],
                "player_name": row["player_name"],
                "team_abbreviation": row["team_abbreviation"],
                "archetype": row["archetype"],
                "cosine_similarity": row["cosine_similarity"],
                "archetype_rank": row["archetype_rank"],
                "supporting_features": _format_feature_list(support, top_n_features, descending=True),
                "gap_features": _format_feature_list(gaps, top_n_features, descending=True),
            }
        )

    return pd.DataFrame(rows).sort_values(["player_name", "archetype_rank"]).reset_index(drop=True)


def _format_feature_list(values: Mapping[str, float], limit: int, descending: bool) -> str:
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=descending)[:limit]
    return "; ".join(f"{feature}={value:.3f}" for feature, value in ordered)

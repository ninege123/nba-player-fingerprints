"""Player-facing summary tables assembled from model outputs."""

from __future__ import annotations

import pandas as pd

from nba_fingerprints.features.schema import validate_columns


def build_player_summary_table(
    features: pd.DataFrame,
    neighbors: pd.DataFrame,
    position_scores: pd.DataFrame,
    archetype_scores: pd.DataFrame,
    archetype_explanations: pd.DataFrame,
) -> pd.DataFrame:
    """Build one compact row per player-season for manual review and reporting."""
    validate_columns(
        features,
        ["season", "player_id", "player_name", "team_abbreviation", "position", "minutes_total", "games_played"],
        dataset_name="player features",
    )
    validate_columns(
        neighbors,
        ["season", "player_id", "team_abbreviation", "neighbor_rank", "neighbor_player_name", "cosine_similarity"],
        dataset_name="neighbors",
    )
    validate_columns(
        position_scores,
        ["season", "player_id", "team_abbreviation", "reference_position", "position_rank", "cosine_similarity"],
        dataset_name="position scores",
    )
    validate_columns(
        archetype_scores,
        ["season", "player_id", "team_abbreviation", "archetype", "archetype_rank", "cosine_similarity"],
        dataset_name="archetype scores",
    )
    validate_columns(
        archetype_explanations,
        ["season", "player_id", "team_abbreviation", "supporting_features", "gap_features"],
        dataset_name="archetype explanations",
    )

    base = features.loc[
        :,
        [
            "season",
            "player_id",
            "player_name",
            "team_abbreviation",
            "position",
            "primary_position",
            "games_played",
            "minutes_total",
            "minutes_per_game",
            "usage_rate",
            "player_impact_estimate",
        ],
    ].copy()

    top_neighbors = neighbors[neighbors["neighbor_rank"] == 1].loc[
        :,
        ["season", "player_id", "team_abbreviation", "neighbor_player_name", "cosine_similarity"],
    ].rename(
        columns={
            "neighbor_player_name": "top_neighbor",
            "cosine_similarity": "top_neighbor_similarity",
        }
    )
    top_positions = position_scores[position_scores["position_rank"] == 1].loc[
        :,
        ["season", "player_id", "team_abbreviation", "reference_position", "cosine_similarity"],
    ].rename(
        columns={
            "reference_position": "top_position_reference",
            "cosine_similarity": "top_position_similarity",
        }
    )
    top_archetypes = archetype_scores[archetype_scores["archetype_rank"] == 1].loc[
        :,
        ["season", "player_id", "team_abbreviation", "archetype", "cosine_similarity"],
    ].rename(
        columns={
            "archetype": "top_archetype",
            "cosine_similarity": "top_archetype_similarity",
        }
    )
    explanations = archetype_explanations.loc[
        :,
        ["season", "player_id", "team_abbreviation", "supporting_features", "gap_features"],
    ]

    summary = (
        base.merge(top_neighbors, on=["season", "player_id", "team_abbreviation"], how="left")
        .merge(top_positions, on=["season", "player_id", "team_abbreviation"], how="left")
        .merge(top_archetypes, on=["season", "player_id", "team_abbreviation"], how="left")
        .merge(explanations, on=["season", "player_id", "team_abbreviation"], how="left")
    )
    return summary.sort_values(["player_name", "team_abbreviation"]).reset_index(drop=True)

"""Build stable app-facing tables from processed fingerprint outputs."""

from __future__ import annotations

import re
from collections.abc import Sequence

import pandas as pd

from nba_fingerprints.features.player_season import FINGERPRINT_FEATURE_COLUMNS
from nba_fingerprints.features.schema import validate_columns
from nba_fingerprints.features.scoring_style import SCORING_STYLE_AVAILABILITY_COLUMNS


CORE_PROFILE_COLUMNS = [
    "season",
    "player_id",
    "player_name",
    "team_abbreviation",
    "position",
    "primary_position",
    "games_played",
    "minutes_total",
    "minutes_per_game",
    "points_per_36",
    "usage_rate",
    "true_shooting_pct",
    "top_neighbor",
    "top_neighbor_similarity",
    "top_archetype",
    "top_archetype_similarity",
    "top_position_reference",
    "top_position_similarity",
    "supporting_features",
    "gap_features",
]

APP_STYLE_COLUMNS = [
    "isolation_frequency",
    "isolation_points_per_possession",
    "pnr_ball_handler_frequency",
    "pnr_ball_handler_points_per_possession",
    "pnr_roll_man_frequency",
    "pnr_roll_man_points_per_possession",
    "spot_up_frequency",
    "spot_up_points_per_possession",
    "cut_frequency",
    "cut_points_per_possession",
    "handoff_frequency",
    "handoff_points_per_possession",
    "post_up_frequency",
    "post_up_points_per_possession",
    "off_screen_frequency",
    "off_screen_points_per_possession",
    "transition_frequency",
    "transition_points_per_possession",
    "rim_attempt_rate",
    "midrange_attempt_rate",
    "corner_three_attempt_rate",
    "above_break_three_attempt_rate",
    "catch_shoot_attempt_frequency",
    "catch_shoot_effective_fg_pct",
    "pull_up_attempt_frequency",
    "pull_up_effective_fg_pct",
    "drives_per_36",
    "frontcourt_touches_per_36",
    "passes_made_per_touch",
    "avg_seconds_per_touch",
]


FEATURE_GROUPS = {
    "minutes_per_game": "role",
    "points_per_36": "scoring",
    "assists_per_36": "playmaking",
    "rebounds_per_36": "rebounding",
    "offensive_rebounds_per_36": "rebounding",
    "defensive_rebounds_per_36": "rebounding",
    "steals_per_36": "defense_proxy",
    "blocks_per_36": "defense_proxy",
    "turnovers_per_36": "playmaking",
    "personal_fouls_per_36": "defense_proxy",
    "three_point_attempt_rate": "shooting",
    "free_throw_attempt_rate": "scoring",
    "effective_fg_pct": "shooting",
    "true_shooting_pct": "scoring",
    "assist_to_turnover": "playmaking",
    "usage_rate": "role",
    "assist_pct": "playmaking",
    "offensive_rebound_pct": "rebounding",
    "defensive_rebound_pct": "rebounding",
    "total_rebound_pct": "rebounding",
    "turnover_pct": "playmaking",
    "pace": "context",
    "player_impact_estimate": "impact",
    "net_rating": "context",
}


HIGHER_IS = {
    "turnovers_per_36": "contextual",
    "personal_fouls_per_36": "contextual",
    "turnover_pct": "contextual",
    "pace": "contextual",
    "net_rating": "better",
}


def build_app_player_profiles(
    features: pd.DataFrame,
    player_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build one compact player table for app search, cards, and profile pages."""
    validate_columns(features, ["season", "player_id", "team_abbreviation", *APP_STYLE_COLUMNS], dataset_name="features")
    validate_columns(player_summary, CORE_PROFILE_COLUMNS, dataset_name="player summary")

    style_columns = [column for column in APP_STYLE_COLUMNS if column in features.columns]
    availability_columns = [column for column in SCORING_STYLE_AVAILABILITY_COLUMNS if column in features.columns]
    style_features = features.loc[:, ["season", "player_id", "team_abbreviation", *style_columns, *availability_columns]]

    profiles = player_summary.loc[:, CORE_PROFILE_COLUMNS].merge(
        style_features,
        on=["season", "player_id", "team_abbreviation"],
        how="left",
    )
    profiles["style_summary"] = profiles.apply(_style_summary, axis=1)
    profiles["peer_typicality_percentile"] = _percentile_rank_by_group(
        profiles,
        group_column="primary_position",
        value_column="top_position_similarity",
    )
    profiles["distinctiveness_score"] = 1.0 - pd.to_numeric(
        profiles["top_neighbor_similarity"],
        errors="coerce",
    ).fillna(0.0)
    return profiles.sort_values(["player_name", "team_abbreviation"]).reset_index(drop=True)


def build_app_similarity_edges(neighbors: pd.DataFrame) -> pd.DataFrame:
    """Build stable nearest-neighbor edges for app graph and comparison views."""
    validate_columns(
        neighbors,
        [
            "season",
            "player_id",
            "player_name",
            "team_abbreviation",
            "neighbor_rank",
            "neighbor_player_id",
            "neighbor_player_name",
            "neighbor_team_abbreviation",
            "cosine_similarity",
        ],
        dataset_name="neighbors",
    )
    output = neighbors.loc[
        :,
        [
            "season",
            "player_id",
            "player_name",
            "team_abbreviation",
            "neighbor_rank",
            "neighbor_player_id",
            "neighbor_player_name",
            "neighbor_team_abbreviation",
            "cosine_similarity",
        ],
    ].copy()
    return output.sort_values(["player_name", "neighbor_rank"]).reset_index(drop=True)


def build_app_feature_metadata(feature_columns: Sequence[str] = FINGERPRINT_FEATURE_COLUMNS) -> pd.DataFrame:
    """Build labels, groups, and display hints for app feature controls."""
    rows = []
    for feature in feature_columns:
        rows.append(
            {
                "feature": feature,
                "label": _feature_label(feature),
                "group": _feature_group(feature),
                "description": _feature_description(feature),
                "higher_is": HIGHER_IS.get(feature, "more"),
                "is_fingerprint_dimension": True,
            }
        )

    for flag in SCORING_STYLE_AVAILABILITY_COLUMNS:
        rows.append(
            {
                "feature": flag,
                "label": _feature_label(flag),
                "group": "source_availability",
                "description": _availability_description(flag),
                "higher_is": "coverage",
                "is_fingerprint_dimension": False,
            }
        )

    return pd.DataFrame(rows)


def _style_summary(row: pd.Series) -> str:
    candidates = [
        ("isolation_frequency", "isolation creator"),
        ("pnr_ball_handler_frequency", "pick-and-roll ball handler"),
        ("pnr_roll_man_frequency", "roll finisher"),
        ("spot_up_frequency", "spot-up scorer"),
        ("off_screen_frequency", "movement shooter"),
        ("cut_frequency", "cutter"),
        ("post_up_frequency", "post scorer"),
        ("transition_frequency", "transition scorer"),
        ("pull_up_attempt_frequency", "pull-up shooter"),
        ("catch_shoot_attempt_frequency", "catch-and-shoot threat"),
        ("rim_attempt_rate", "rim attacker"),
        ("above_break_three_attempt_rate", "above-break shooter"),
    ]
    values = []
    for feature, label in candidates:
        if feature in row and pd.notna(row[feature]):
            values.append((float(row[feature]), label))
    top_labels = [label for value, label in sorted(values, reverse=True)[:3] if value > 0]
    return ", ".join(top_labels)


def _percentile_rank_by_group(frame: pd.DataFrame, group_column: str, value_column: str) -> pd.Series:
    values = pd.to_numeric(frame[value_column], errors="coerce")
    if group_column not in frame.columns:
        return values.rank(pct=True).fillna(0.0)
    return values.groupby(frame[group_column]).rank(pct=True).fillna(0.0)


def _feature_group(feature: str) -> str:
    if feature in FEATURE_GROUPS:
        return FEATURE_GROUPS[feature]
    if _has_any(feature, ["isolation", "pnr", "spot_up", "cut", "handoff", "post_up", "off_screen", "putback", "transition"]):
        return "play_type"
    if _has_any(feature, ["rim_", "paint_", "midrange", "corner_three", "above_break", "restricted_area"]):
        return "shot_location"
    if _has_any(feature, ["catch_shoot", "pull_up"]):
        return "tracking_shot"
    if _has_any(feature, ["drives", "touches", "passes", "seconds_per_touch"]):
        return "touch_tracking"
    return "other"


def _feature_label(feature: str) -> str:
    replacements = {
        "pct": "%",
        "fg": "FG",
        "efg": "eFG",
        "ppp": "PPP",
        "pnr": "P&R",
        "ra": "RA",
    }
    label = feature.replace("_", " ")
    for source, replacement in replacements.items():
        label = re.sub(rf"\b{source}\b", replacement, label, flags=re.IGNORECASE)
    return label.title().replace("P&R", "P&R").replace("FG", "FG").replace("Efg", "eFG")


def _feature_description(feature: str) -> str:
    label = _feature_label(feature)
    group = _feature_group(feature).replace("_", " ")
    return f"{label} from the {group} feature group."


def _availability_description(feature: str) -> str:
    descriptions = {
        "has_synergy_data": "Whether the player appears in at least one offensive Synergy play-type table.",
        "has_shot_location_data": "Whether the player appears in the shot-location source table.",
        "has_tracking_shot_data": "Whether the player appears in catch-and-shoot or pull-up tracking tables.",
        "has_touch_tracking_data": "Whether the player appears in drive, touch, passing, paint-touch, elbow-touch, or post-touch tracking tables.",
    }
    return descriptions[feature]


def _has_any(value: str, tokens: list[str]) -> bool:
    return any(token in value for token in tokens)

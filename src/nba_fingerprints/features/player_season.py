"""Player-season feature engineering from NBA box-score totals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_fingerprints.features.schema import validate_columns


PLAYER_SEASON_RAW_COLUMNS = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "AGE",
    "GP",
    "MIN",
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FTA",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "TOV",
    "STL",
    "BLK",
    "PF",
    "PTS",
]

FINGERPRINT_FEATURE_COLUMNS = [
    "minutes_per_game",
    "points_per_36",
    "assists_per_36",
    "rebounds_per_36",
    "offensive_rebounds_per_36",
    "defensive_rebounds_per_36",
    "steals_per_36",
    "blocks_per_36",
    "turnovers_per_36",
    "personal_fouls_per_36",
    "three_point_attempt_rate",
    "free_throw_attempt_rate",
    "effective_fg_pct",
    "true_shooting_pct",
    "assist_to_turnover",
]


def build_player_season_features(
    raw_stats: pd.DataFrame,
    season: str,
    min_minutes: float = 0.0,
) -> pd.DataFrame:
    """Build first-pass fingerprint features from player-season box-score totals."""
    validate_columns(raw_stats, PLAYER_SEASON_RAW_COLUMNS, dataset_name="player season raw stats")

    frame = raw_stats.copy()
    numeric_columns = [column for column in PLAYER_SEASON_RAW_COLUMNS if column not in {"PLAYER_NAME", "TEAM_ABBREVIATION"}]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if min_minutes > 0:
        frame = frame[frame["MIN"] >= min_minutes].copy()

    features = pd.DataFrame(
        {
            "season": season,
            "player_id": frame["PLAYER_ID"].astype("Int64"),
            "player_name": frame["PLAYER_NAME"],
            "team_id": frame["TEAM_ID"].astype("Int64"),
            "team_abbreviation": frame["TEAM_ABBREVIATION"],
            "age": frame["AGE"],
            "games_played": frame["GP"],
            "minutes_total": frame["MIN"],
            "minutes_per_game": _safe_divide(frame["MIN"], frame["GP"]),
            "points_per_36": _per_36(frame["PTS"], frame["MIN"]),
            "assists_per_36": _per_36(frame["AST"], frame["MIN"]),
            "rebounds_per_36": _per_36(frame["REB"], frame["MIN"]),
            "offensive_rebounds_per_36": _per_36(frame["OREB"], frame["MIN"]),
            "defensive_rebounds_per_36": _per_36(frame["DREB"], frame["MIN"]),
            "steals_per_36": _per_36(frame["STL"], frame["MIN"]),
            "blocks_per_36": _per_36(frame["BLK"], frame["MIN"]),
            "turnovers_per_36": _per_36(frame["TOV"], frame["MIN"]),
            "personal_fouls_per_36": _per_36(frame["PF"], frame["MIN"]),
            "three_point_attempt_rate": _safe_divide(frame["FG3A"], frame["FGA"]),
            "free_throw_attempt_rate": _safe_divide(frame["FTA"], frame["FGA"]),
            "effective_fg_pct": _safe_divide(frame["FGM"] + 0.5 * frame["FG3M"], frame["FGA"]),
            "true_shooting_pct": _safe_divide(frame["PTS"], 2 * (frame["FGA"] + 0.44 * frame["FTA"])),
            "assist_to_turnover": _safe_divide(frame["AST"], frame["TOV"]),
        }
    )

    return features.reset_index(drop=True)


def _per_36(numerator: pd.Series, minutes: pd.Series) -> pd.Series:
    return _safe_divide(numerator * 36.0, minutes)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.astype(float).divide(denominator.astype(float).replace(0, np.nan))
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

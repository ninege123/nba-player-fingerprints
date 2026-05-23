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

PLAYER_SEASON_ADVANCED_COLUMNS = [
    "PLAYER_ID",
    "TEAM_ID",
    "AST_PCT",
    "OREB_PCT",
    "DREB_PCT",
    "REB_PCT",
    "TM_TOV_PCT",
    "TS_PCT",
    "USG_PCT",
    "PACE",
    "PIE",
    "NET_RATING",
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
    "usage_rate",
    "assist_pct",
    "offensive_rebound_pct",
    "defensive_rebound_pct",
    "total_rebound_pct",
    "turnover_pct",
    "pace",
    "player_impact_estimate",
    "net_rating",
]


def build_player_season_features(
    raw_stats: pd.DataFrame,
    season: str,
    advanced_stats: pd.DataFrame | None = None,
    min_minutes: float = 0.0,
) -> pd.DataFrame:
    """Build first-pass fingerprint features from player-season box-score totals."""
    validate_columns(raw_stats, PLAYER_SEASON_RAW_COLUMNS, dataset_name="player season raw stats")

    frame = raw_stats.copy()
    if advanced_stats is not None:
        frame = _attach_advanced_stats(frame, advanced_stats)

    numeric_columns = [column for column in PLAYER_SEASON_RAW_COLUMNS if column not in {"PLAYER_NAME", "TEAM_ABBREVIATION"}]
    numeric_columns.extend([column for column in _advanced_output_columns() if column in frame.columns])
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
            "usage_rate": _advanced_or_zero(frame, "USG_PCT"),
            "assist_pct": _advanced_or_zero(frame, "AST_PCT"),
            "offensive_rebound_pct": _advanced_or_zero(frame, "OREB_PCT"),
            "defensive_rebound_pct": _advanced_or_zero(frame, "DREB_PCT"),
            "total_rebound_pct": _advanced_or_zero(frame, "REB_PCT"),
            "turnover_pct": _advanced_or_zero(frame, "TM_TOV_PCT"),
            "pace": _advanced_or_zero(frame, "PACE"),
            "player_impact_estimate": _advanced_or_zero(frame, "PIE"),
            "net_rating": _advanced_or_zero(frame, "NET_RATING"),
        }
    )

    return features.reset_index(drop=True)


def _per_36(numerator: pd.Series, minutes: pd.Series) -> pd.Series:
    return _safe_divide(numerator * 36.0, minutes)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.astype(float).divide(denominator.astype(float).replace(0, np.nan))
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _attach_advanced_stats(raw_stats: pd.DataFrame, advanced_stats: pd.DataFrame) -> pd.DataFrame:
    validate_columns(advanced_stats, PLAYER_SEASON_ADVANCED_COLUMNS, dataset_name="player season advanced stats")
    advanced = advanced_stats.loc[:, PLAYER_SEASON_ADVANCED_COLUMNS].drop_duplicates(subset=["PLAYER_ID", "TEAM_ID"])
    return raw_stats.merge(advanced, on=["PLAYER_ID", "TEAM_ID"], how="left")


def _advanced_output_columns() -> list[str]:
    return [column for column in PLAYER_SEASON_ADVANCED_COLUMNS if column not in {"PLAYER_ID", "TEAM_ID"}]


def _advanced_or_zero(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

"""NBA Stats API client helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import (
    leaguedashplayerptshot,
    leaguedashplayershotlocations,
    leaguedashplayerstats,
    leaguedashptstats,
    playerindex,
    synergyplaytypes,
)

from nba_fingerprints.data.cache import cache_path, cached_frame_exists, read_cached_frame, write_cached_frame
from nba_fingerprints.features.schema import validate_columns


PLAYER_STATS_REQUIRED_COLUMNS = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "AGE",
    "GP",
    "MIN",
]

PLAYER_INDEX_REQUIRED_COLUMNS = [
    "PERSON_ID",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "POSITION",
]

PLAYER_ID_REQUIRED_COLUMNS = [
    "PLAYER_ID",
    "PLAYER_NAME",
]

SYNERGY_PLAY_TYPES = {
    "isolation": "Isolation",
    "pnr_ball_handler": "PRBallHandler",
    "pnr_roll_man": "PRRollman",
    "spot_up": "Spotup",
    "cut": "Cut",
    "handoff": "Handoff",
    "post_up": "Postup",
    "off_screen": "OffScreen",
    "putback": "OffRebound",
    "transition": "Transition",
}

TRACKING_MEASURE_TYPES = {
    "catch_shoot": "CatchShoot",
    "pull_up": "PullUpShot",
    "drives": "Drives",
    "passing": "Passing",
    "possessions": "Possessions",
    "paint_touches": "PaintTouch",
    "post_touches": "PostTouch",
    "elbow_touches": "ElbowTouch",
}


def fetch_player_season_stats(
    season: str,
    measure_type: str = "Base",
    per_mode: str = "Totals",
    season_type: str = "Regular Season",
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch player-season stats from the NBA Stats league dash endpoint."""
    endpoint = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense=measure_type,
        per_mode_detailed=per_mode,
        season_type_all_star=season_type,
        timeout=timeout,
    )
    frame = endpoint.get_data_frames()[0]
    validate_columns(frame, PLAYER_STATS_REQUIRED_COLUMNS, dataset_name="player season stats")
    return frame


def load_player_season_stats(
    season: str,
    measure_type: str = "Base",
    per_mode: str = "Totals",
    season_type: str = "Regular Season",
    cache_dir: Path | str = "data/raw",
    use_cache: bool = True,
    file_format: str = "parquet",
    timeout: int = 30,
) -> pd.DataFrame:
    """Load player-season stats from cache or the NBA Stats API."""
    dataset_name = f"player_season_stats_{measure_type.lower()}_{per_mode.lower().replace(' ', '_')}"
    path = cache_path(dataset_name, season, cache_dir=cache_dir, file_format=file_format)

    if use_cache and cached_frame_exists(path):
        return read_cached_frame(path)

    frame = fetch_player_season_stats(
        season=season,
        measure_type=measure_type,
        per_mode=per_mode,
        season_type=season_type,
        timeout=timeout,
    )
    write_cached_frame(frame, path)
    return frame


def fetch_player_synergy_play_type(
    season: str,
    play_type: str,
    per_mode: str = "Totals",
    season_type: str = "Regular Season",
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch player offensive play-type stats from the NBA Stats Synergy endpoint."""
    endpoint = synergyplaytypes.SynergyPlayTypes(
        season=season,
        play_type_nullable=_resolve_lookup(SYNERGY_PLAY_TYPES, play_type, "play_type"),
        type_grouping_nullable="offensive",
        player_or_team_abbreviation="P",
        per_mode_simple=per_mode,
        season_type_all_star=season_type,
        timeout=timeout,
    )
    frame = endpoint.get_data_frames()[0]
    validate_columns(frame, PLAYER_ID_REQUIRED_COLUMNS, dataset_name=f"{play_type} Synergy stats")
    return frame


def load_player_synergy_play_type(
    season: str,
    play_type: str,
    per_mode: str = "Totals",
    season_type: str = "Regular Season",
    cache_dir: Path | str = "data/raw",
    use_cache: bool = True,
    file_format: str = "parquet",
    timeout: int = 30,
) -> pd.DataFrame:
    """Load one player offensive play-type table from cache or the NBA Stats API."""
    play_type_key = _normalize_key(play_type)
    _resolve_lookup(SYNERGY_PLAY_TYPES, play_type_key, "play_type")
    dataset_name = f"player_synergy_{play_type_key}"
    path = cache_path(dataset_name, season, cache_dir=cache_dir, file_format=file_format)

    if use_cache and cached_frame_exists(path):
        return read_cached_frame(path)

    frame = fetch_player_synergy_play_type(
        season=season,
        play_type=play_type_key,
        per_mode=per_mode,
        season_type=season_type,
        timeout=timeout,
    )
    write_cached_frame(frame, path)
    return frame


def fetch_player_shot_locations(
    season: str,
    per_mode: str = "Totals",
    season_type: str = "Regular Season",
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch player shot-location summaries from the NBA Stats API."""
    endpoint = leaguedashplayershotlocations.LeagueDashPlayerShotLocations(
        season=season,
        per_mode_detailed=per_mode,
        season_type_all_star=season_type,
        timeout=timeout,
    )
    frame = _flatten_columns(endpoint.get_data_frames()[0])
    validate_columns(frame, PLAYER_ID_REQUIRED_COLUMNS, dataset_name="player shot locations")
    return frame


def load_player_shot_locations(
    season: str,
    per_mode: str = "Totals",
    season_type: str = "Regular Season",
    cache_dir: Path | str = "data/raw",
    use_cache: bool = True,
    file_format: str = "parquet",
    timeout: int = 30,
) -> pd.DataFrame:
    """Load player shot-location summaries from cache or the NBA Stats API."""
    path = cache_path("player_shot_locations", season, cache_dir=cache_dir, file_format=file_format)

    if use_cache and cached_frame_exists(path):
        return read_cached_frame(path)

    frame = fetch_player_shot_locations(
        season=season,
        per_mode=per_mode,
        season_type=season_type,
        timeout=timeout,
    )
    write_cached_frame(frame, path)
    return frame


def fetch_player_tracking_stats(
    season: str,
    measure_type: str,
    per_mode: str = "Totals",
    season_type: str = "Regular Season",
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch player tracking summaries from the NBA Stats API."""
    endpoint = leaguedashptstats.LeagueDashPtStats(
        season=season,
        pt_measure_type=_resolve_lookup(TRACKING_MEASURE_TYPES, measure_type, "measure_type"),
        player_or_team="Player",
        per_mode_simple=per_mode,
        season_type_all_star=season_type,
        timeout=timeout,
    )
    frame = endpoint.get_data_frames()[0]
    validate_columns(frame, PLAYER_ID_REQUIRED_COLUMNS, dataset_name=f"{measure_type} tracking stats")
    return frame


def load_player_tracking_stats(
    season: str,
    measure_type: str,
    per_mode: str = "Totals",
    season_type: str = "Regular Season",
    cache_dir: Path | str = "data/raw",
    use_cache: bool = True,
    file_format: str = "parquet",
    timeout: int = 30,
) -> pd.DataFrame:
    """Load one player tracking summary table from cache or the NBA Stats API."""
    measure_type_key = _normalize_key(measure_type)
    _resolve_lookup(TRACKING_MEASURE_TYPES, measure_type_key, "measure_type")
    dataset_name = f"player_tracking_{measure_type_key}"
    path = cache_path(dataset_name, season, cache_dir=cache_dir, file_format=file_format)

    if use_cache and cached_frame_exists(path):
        return read_cached_frame(path)

    frame = fetch_player_tracking_stats(
        season=season,
        measure_type=measure_type_key,
        per_mode=per_mode,
        season_type=season_type,
        timeout=timeout,
    )
    write_cached_frame(frame, path)
    return frame


def fetch_player_tracking_shot_profile(
    season: str,
    per_mode: str = "Totals",
    season_type: str = "Regular Season",
    timeout: int = 30,
    **filters: str,
) -> pd.DataFrame:
    """Fetch filtered player shot-tracking summaries such as dribble or touch-time splits."""
    endpoint = leaguedashplayerptshot.LeagueDashPlayerPtShot(
        season=season,
        per_mode_simple=per_mode,
        season_type_all_star=season_type,
        timeout=timeout,
        **filters,
    )
    frame = endpoint.get_data_frames()[0]
    validate_columns(frame, PLAYER_ID_REQUIRED_COLUMNS, dataset_name="player tracking shot profile")
    return frame


def load_player_tracking_shot_profile(
    season: str,
    profile_name: str,
    per_mode: str = "Totals",
    season_type: str = "Regular Season",
    cache_dir: Path | str = "data/raw",
    use_cache: bool = True,
    file_format: str = "parquet",
    timeout: int = 30,
    **filters: str,
) -> pd.DataFrame:
    """Load a filtered player shot-tracking summary from cache or the NBA Stats API."""
    dataset_name = f"player_tracking_shot_{_normalize_key(profile_name)}"
    path = cache_path(dataset_name, season, cache_dir=cache_dir, file_format=file_format)

    if use_cache and cached_frame_exists(path):
        return read_cached_frame(path)

    frame = fetch_player_tracking_shot_profile(
        season=season,
        per_mode=per_mode,
        season_type=season_type,
        timeout=timeout,
        **filters,
    )
    write_cached_frame(frame, path)
    return frame


def fetch_player_index(
    season: str,
    historical: str = "1",
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch NBA player index metadata, including listed positions."""
    endpoint = playerindex.PlayerIndex(season=season, historical_nullable=historical, timeout=timeout)
    frame = endpoint.get_data_frames()[0]
    validate_columns(frame, PLAYER_INDEX_REQUIRED_COLUMNS, dataset_name="player index")
    return frame


def load_player_index(
    season: str,
    cache_dir: Path | str = "data/raw",
    use_cache: bool = True,
    file_format: str = "parquet",
    timeout: int = 30,
) -> pd.DataFrame:
    """Load player index metadata from cache or the NBA Stats API."""
    path = cache_path("player_index_historical", season, cache_dir=cache_dir, file_format=file_format)

    if use_cache and cached_frame_exists(path):
        return read_cached_frame(path)

    frame = fetch_player_index(season=season, timeout=timeout)
    write_cached_frame(frame, path)
    return frame


def _normalize_key(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _resolve_lookup(values: dict[str, str], key: str, parameter_name: str) -> str:
    normalized_key = _normalize_key(key)
    if normalized_key not in values:
        raise ValueError(f"Unknown {parameter_name}: {key}")
    return values[normalized_key]


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    current = frame.copy()
    current.columns = [
        " ".join(str(part) for part in column if str(part) and str(part) != "nan")
        if isinstance(column, tuple)
        else str(column)
        for column in current.columns
    ]
    return current

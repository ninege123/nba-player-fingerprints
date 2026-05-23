"""NBA Stats API client helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats, playerindex

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

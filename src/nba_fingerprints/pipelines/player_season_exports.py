"""Export player-season feature, fingerprint, and neighbor tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from nba_fingerprints.data.nba_api_client import load_player_index, load_player_season_stats
from nba_fingerprints.features.archetypes import build_archetype_references, score_archetype_similarity
from nba_fingerprints.features.fingerprints import build_fingerprint_table
from nba_fingerprints.features.player_season import build_player_season_features
from nba_fingerprints.features.positions import attach_position_labels, build_position_references, score_position_similarity
from nba_fingerprints.models.neighbors import find_nearest_neighbors


PROCESSED_DATA_DIR = Path("data/processed")


@dataclass(frozen=True)
class PlayerSeasonExportPaths:
    """Output paths produced by the player-season export pipeline."""

    features: Path
    fingerprints: Path
    neighbors: Path
    position_references: Path
    position_scores: Path
    archetype_references: Path
    archetype_scores: Path


def player_season_export_paths(
    season: str,
    output_dir: Path | str = PROCESSED_DATA_DIR,
    file_format: str = "csv",
) -> PlayerSeasonExportPaths:
    """Build stable output paths for processed player-season tables."""
    if file_format not in {"csv", "parquet"}:
        raise ValueError("file_format must be 'csv' or 'parquet'")

    safe_season = season.replace("-", "_")
    output_path = Path(output_dir)
    suffix = file_format
    return PlayerSeasonExportPaths(
        features=output_path / f"player_season_features_{safe_season}.{suffix}",
        fingerprints=output_path / f"player_fingerprints_{safe_season}.{suffix}",
        neighbors=output_path / f"player_neighbors_{safe_season}.{suffix}",
        position_references=output_path / f"position_references_{safe_season}.{suffix}",
        position_scores=output_path / f"player_position_scores_{safe_season}.{suffix}",
        archetype_references=output_path / f"archetype_references_{safe_season}.{suffix}",
        archetype_scores=output_path / f"player_archetype_scores_{safe_season}.{suffix}",
    )


def build_player_season_export_tables(
    raw_stats: pd.DataFrame,
    season: str,
    player_index: pd.DataFrame | None = None,
    min_minutes: float = 500.0,
    top_n: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build processed feature, fingerprint, and nearest-neighbor tables."""
    features = build_player_season_features(raw_stats, season=season, min_minutes=min_minutes)
    if player_index is not None:
        features = attach_position_labels(features, player_index)

    id_columns = ["season", "player_id", "player_name", "team_abbreviation"]
    if "position" in features.columns and "primary_position" in features.columns:
        id_columns.extend(["position", "primary_position"])

    fingerprints = build_fingerprint_table(features, id_columns=id_columns)
    neighbors = find_nearest_neighbors(fingerprints, top_n=top_n)
    position_references = build_position_references(fingerprints) if "primary_position" in fingerprints.columns else pd.DataFrame()
    position_scores = (
        score_position_similarity(fingerprints, position_references)
        if not position_references.empty and "primary_position" in fingerprints.columns
        else pd.DataFrame()
    )
    archetype_references = build_archetype_references()
    archetype_scores = score_archetype_similarity(fingerprints, archetype_references)
    return features, fingerprints, neighbors, position_references, position_scores, archetype_references, archetype_scores


def export_player_season_tables(
    season: str,
    min_minutes: float = 500.0,
    top_n: int = 5,
    output_dir: Path | str = PROCESSED_DATA_DIR,
    raw_cache_dir: Path | str = "data/raw",
    file_format: str = "csv",
    use_cache: bool = True,
) -> PlayerSeasonExportPaths:
    """Load raw stats, build processed tables, and write them to disk."""
    raw_stats = load_player_season_stats(
        season=season,
        cache_dir=raw_cache_dir,
        use_cache=use_cache,
        file_format=file_format,
    )
    player_index = load_player_index(
        season=season,
        cache_dir=raw_cache_dir,
        use_cache=use_cache,
        file_format=file_format,
    )
    (
        features,
        fingerprints,
        neighbors,
        position_references,
        position_scores,
        archetype_references,
        archetype_scores,
    ) = build_player_season_export_tables(
        raw_stats,
        season=season,
        player_index=player_index,
        min_minutes=min_minutes,
        top_n=top_n,
    )

    paths = player_season_export_paths(season, output_dir=output_dir, file_format=file_format)
    _write_frame(features, paths.features)
    _write_frame(fingerprints, paths.fingerprints)
    _write_frame(neighbors, paths.neighbors)
    _write_frame(position_references, paths.position_references)
    _write_frame(position_scores, paths.position_scores)
    _write_frame(archetype_references, paths.archetype_references)
    _write_frame(archetype_scores, paths.archetype_scores)
    return paths


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        raise ValueError("Output path must end in .csv or .parquet")

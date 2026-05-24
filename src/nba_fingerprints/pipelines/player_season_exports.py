"""Export player-season feature, fingerprint, and neighbor tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from nba_fingerprints.data.nba_api_client import (
    SYNERGY_PLAY_TYPES,
    TRACKING_MEASURE_TYPES,
    load_player_index,
    load_player_season_stats,
    load_player_shot_locations,
    load_player_synergy_play_type,
    load_player_tracking_stats,
)
from nba_fingerprints.features.archetypes import (
    build_archetype_references,
    explain_top_archetype_matches,
    score_archetype_similarity,
)
from nba_fingerprints.features.fingerprints import build_fingerprint_table
from nba_fingerprints.features.player_season import build_player_season_features
from nba_fingerprints.features.positions import attach_position_labels, build_position_references, score_position_similarity
from nba_fingerprints.features.scoring_style import attach_scoring_style_features
from nba_fingerprints.models.neighbors import find_nearest_neighbors
from nba_fingerprints.models.summaries import build_player_summary_table


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
    archetype_explanations: Path
    player_summary: Path


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
        archetype_explanations=output_path / f"player_archetype_explanations_{safe_season}.{suffix}",
        player_summary=output_path / f"player_summary_{safe_season}.{suffix}",
    )


def build_player_season_export_tables(
    raw_stats: pd.DataFrame,
    season: str,
    advanced_stats: pd.DataFrame | None = None,
    player_index: pd.DataFrame | None = None,
    synergy_frames: dict[str, pd.DataFrame] | None = None,
    shot_locations: pd.DataFrame | None = None,
    tracking_frames: dict[str, pd.DataFrame] | None = None,
    min_minutes: float = 500.0,
    top_n: int = 5,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build processed feature, fingerprint, and nearest-neighbor tables."""
    features = build_player_season_features(
        raw_stats,
        season=season,
        advanced_stats=advanced_stats,
        min_minutes=min_minutes,
    )
    if player_index is not None:
        features = attach_position_labels(features, player_index)
    features = attach_scoring_style_features(
        features,
        synergy_frames=synergy_frames,
        shot_locations=shot_locations,
        tracking_frames=tracking_frames,
    )

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
    archetype_explanations = explain_top_archetype_matches(fingerprints, archetype_references, archetype_scores)
    player_summary = build_player_summary_table(
        features,
        neighbors,
        position_scores,
        archetype_scores,
        archetype_explanations,
    )
    return (
        features,
        fingerprints,
        neighbors,
        position_references,
        position_scores,
        archetype_references,
        archetype_scores,
        archetype_explanations,
        player_summary,
    )


def export_player_season_tables(
    season: str,
    min_minutes: float = 500.0,
    top_n: int = 5,
    output_dir: Path | str = PROCESSED_DATA_DIR,
    raw_cache_dir: Path | str = "data/raw",
    file_format: str = "csv",
    use_cache: bool = True,
    include_scoring_style: bool = False,
    ignore_scoring_style_errors: bool = True,
) -> PlayerSeasonExportPaths:
    """Load raw stats, build processed tables, and write them to disk."""
    raw_stats = load_player_season_stats(
        season=season,
        cache_dir=raw_cache_dir,
        use_cache=use_cache,
        file_format=file_format,
    )
    advanced_stats = load_player_season_stats(
        season=season,
        measure_type="Advanced",
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
    synergy_frames: dict[str, pd.DataFrame] | None = None
    shot_locations: pd.DataFrame | None = None
    tracking_frames: dict[str, pd.DataFrame] | None = None
    if include_scoring_style:
        synergy_frames, shot_locations, tracking_frames = _load_scoring_style_sources(
            season=season,
            raw_cache_dir=raw_cache_dir,
            file_format=file_format,
            use_cache=use_cache,
            ignore_errors=ignore_scoring_style_errors,
        )
    (
        features,
        fingerprints,
        neighbors,
        position_references,
        position_scores,
        archetype_references,
        archetype_scores,
        archetype_explanations,
        player_summary,
    ) = build_player_season_export_tables(
        raw_stats,
        season=season,
        advanced_stats=advanced_stats,
        player_index=player_index,
        synergy_frames=synergy_frames,
        shot_locations=shot_locations,
        tracking_frames=tracking_frames,
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
    _write_frame(archetype_explanations, paths.archetype_explanations)
    _write_frame(player_summary, paths.player_summary)
    return paths


def _load_scoring_style_sources(
    season: str,
    raw_cache_dir: Path | str,
    file_format: str,
    use_cache: bool,
    ignore_errors: bool,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None, dict[str, pd.DataFrame]]:
    synergy_frames: dict[str, pd.DataFrame] = {}
    tracking_frames: dict[str, pd.DataFrame] = {}

    for play_type in SYNERGY_PLAY_TYPES:
        frame = _load_optional_source(
            load_player_synergy_play_type,
            ignore_errors=ignore_errors,
            season=season,
            play_type=play_type,
            cache_dir=raw_cache_dir,
            use_cache=use_cache,
            file_format=file_format,
        )
        if frame is not None:
            synergy_frames[play_type] = frame

    shot_locations = _load_optional_source(
        load_player_shot_locations,
        ignore_errors=ignore_errors,
        season=season,
        cache_dir=raw_cache_dir,
        use_cache=use_cache,
        file_format=file_format,
    )

    for measure_type in TRACKING_MEASURE_TYPES:
        frame = _load_optional_source(
            load_player_tracking_stats,
            ignore_errors=ignore_errors,
            season=season,
            measure_type=measure_type,
            cache_dir=raw_cache_dir,
            use_cache=use_cache,
            file_format=file_format,
        )
        if frame is not None:
            tracking_frames[measure_type] = frame

    return synergy_frames, shot_locations, tracking_frames


def _load_optional_source(loader, ignore_errors: bool, **kwargs) -> pd.DataFrame | None:
    try:
        return loader(**kwargs)
    except Exception:
        if ignore_errors:
            return None
        raise


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        raise ValueError("Output path must end in .csv or .parquet")

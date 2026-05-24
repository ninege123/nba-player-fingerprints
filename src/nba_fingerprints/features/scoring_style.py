"""Optional scoring-style features from NBA tracking, shot-location, and play-type tables."""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np
import pandas as pd

from nba_fingerprints.features.schema import validate_columns


SYNERGY_FEATURE_PREFIXES = {
    "isolation": "isolation",
    "pnr_ball_handler": "pnr_ball_handler",
    "pnr_roll_man": "pnr_roll_man",
    "spot_up": "spot_up",
    "cut": "cut",
    "handoff": "handoff",
    "post_up": "post_up",
    "off_screen": "off_screen",
    "putback": "putback",
    "transition": "transition",
}

SYNERGY_FEATURE_COLUMNS = [
    f"{prefix}_{suffix}"
    for prefix in SYNERGY_FEATURE_PREFIXES.values()
    for suffix in [
        "frequency",
        "points_per_possession",
        "score_frequency",
        "turnover_frequency",
        "effective_fg_pct",
        "percentile",
    ]
]

SHOT_LOCATION_FEATURE_COLUMNS = [
    "rim_attempt_rate",
    "paint_non_ra_attempt_rate",
    "midrange_attempt_rate",
    "corner_three_attempt_rate",
    "above_break_three_attempt_rate",
    "restricted_area_fg_pct",
    "corner_three_fg_pct",
]

TRACKING_FEATURE_COLUMNS = [
    "catch_shoot_attempt_frequency",
    "catch_shoot_effective_fg_pct",
    "pull_up_attempt_frequency",
    "pull_up_effective_fg_pct",
    "drives_per_36",
    "paint_touches_per_36",
    "post_touches_per_36",
    "elbow_touches_per_36",
    "frontcourt_touches_per_36",
    "passes_made_per_touch",
    "avg_seconds_per_touch",
]

SCORING_STYLE_FEATURE_COLUMNS = [
    *SYNERGY_FEATURE_COLUMNS,
    *SHOT_LOCATION_FEATURE_COLUMNS,
    *TRACKING_FEATURE_COLUMNS,
]

SCORING_STYLE_AVAILABILITY_COLUMNS = [
    "has_synergy_data",
    "has_shot_location_data",
    "has_tracking_shot_data",
    "has_touch_tracking_data",
]


def attach_scoring_style_features(
    player_features: pd.DataFrame,
    synergy_frames: Mapping[str, pd.DataFrame] | None = None,
    shot_locations: pd.DataFrame | None = None,
    tracking_frames: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Attach optional scoring-style features to an existing player-season feature table."""
    scoring_features = build_scoring_style_features(
        player_features,
        synergy_frames=synergy_frames,
        shot_locations=shot_locations,
        tracking_frames=tracking_frames,
    )
    passthrough = player_features.drop(
        columns=[*SCORING_STYLE_FEATURE_COLUMNS, *SCORING_STYLE_AVAILABILITY_COLUMNS],
        errors="ignore",
    )
    merged = passthrough.merge(scoring_features, on="player_id", how="left")
    fill_values = {column: 0.0 for column in SCORING_STYLE_FEATURE_COLUMNS}
    fill_values.update({column: False for column in SCORING_STYLE_AVAILABILITY_COLUMNS})
    return merged.fillna(fill_values)


def build_scoring_style_features(
    player_features: pd.DataFrame,
    synergy_frames: Mapping[str, pd.DataFrame] | None = None,
    shot_locations: pd.DataFrame | None = None,
    tracking_frames: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build scoring-style columns keyed by `player_id` from optional source tables."""
    validate_columns(player_features, ["player_id"], dataset_name="player features")
    output = pd.DataFrame({"player_id": pd.to_numeric(player_features["player_id"], errors="coerce").astype("Int64")})
    for column in SCORING_STYLE_FEATURE_COLUMNS:
        output[column] = 0.0
    for column in SCORING_STYLE_AVAILABILITY_COLUMNS:
        output[column] = False

    if synergy_frames:
        output = _merge_feature_block(output, _build_synergy_features(synergy_frames))
        output["has_synergy_data"] = output["has_synergy_data"] | output["player_id"].isin(
            _source_player_ids(synergy_frames.values())
        )
    if shot_locations is not None:
        output = _merge_feature_block(output, _build_shot_location_features(shot_locations))
        output["has_shot_location_data"] = output["has_shot_location_data"] | output["player_id"].isin(
            _source_player_ids([shot_locations])
        )
    if tracking_frames:
        output = _merge_feature_block(output, _build_tracking_features(player_features, tracking_frames))
        output["has_tracking_shot_data"] = output["has_tracking_shot_data"] | output["player_id"].isin(
            _source_player_ids(
                frame
                for measure_type, frame in tracking_frames.items()
                if _normalize_key(measure_type) in {"catch_shoot", "pull_up", "pull_up_shot"}
            )
        )
        output["has_touch_tracking_data"] = output["has_touch_tracking_data"] | output["player_id"].isin(
            _source_player_ids(
                frame
                for measure_type, frame in tracking_frames.items()
                if _normalize_key(measure_type)
                in {"drives", "passing", "possessions", "paint_touches", "post_touches", "elbow_touches"}
            )
        )

    fill_values = {column: 0.0 for column in SCORING_STYLE_FEATURE_COLUMNS}
    fill_values.update({column: False for column in SCORING_STYLE_AVAILABILITY_COLUMNS})
    return output.loc[:, ["player_id", *SCORING_STYLE_FEATURE_COLUMNS, *SCORING_STYLE_AVAILABILITY_COLUMNS]].fillna(
        fill_values
    )


def _build_synergy_features(synergy_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    for play_type, frame in synergy_frames.items():
        prefix = SYNERGY_FEATURE_PREFIXES.get(_normalize_key(play_type))
        if prefix is None or frame.empty:
            continue

        current = _flatten_columns(frame)
        validate_columns(current, ["PLAYER_ID"], dataset_name=f"{play_type} Synergy stats")
        block = pd.DataFrame({"player_id": _numeric(current["PLAYER_ID"]).astype("Int64")})
        block[f"{prefix}_frequency"] = _optional_numeric(current, "POSS_PCT")
        block[f"{prefix}_points_per_possession"] = _optional_numeric(current, "PPP")
        block[f"{prefix}_score_frequency"] = _optional_numeric(current, "SCORE_POSS_PCT")
        block[f"{prefix}_turnover_frequency"] = _optional_numeric(current, "TOV_POSS_PCT")
        block[f"{prefix}_effective_fg_pct"] = _optional_numeric(current, "EFG_PCT")
        block[f"{prefix}_percentile"] = _optional_numeric(current, "PERCENTILE")
        blocks.append(block)

    return _combine_blocks(blocks)


def _build_shot_location_features(shot_locations: pd.DataFrame) -> pd.DataFrame:
    current = _flatten_columns(shot_locations)
    validate_columns(current, ["PLAYER_ID"], dataset_name="player shot locations")

    restricted_fga = _zone_numeric(current, "Restricted Area", "FGA")
    paint_fga = _zone_numeric(current, "In The Paint (Non-RA)", "FGA")
    midrange_fga = _zone_numeric(current, "Mid-Range", "FGA")
    left_corner_fga = _zone_numeric(current, "Left Corner 3", "FGA")
    right_corner_fga = _zone_numeric(current, "Right Corner 3", "FGA")
    above_break_fga = _zone_numeric(current, "Above the Break 3", "FGA")
    backcourt_fga = _zone_numeric(current, "Backcourt", "FGA")
    total_fga = restricted_fga + paint_fga + midrange_fga + left_corner_fga + right_corner_fga + above_break_fga + backcourt_fga

    left_corner_fgm = _zone_numeric(current, "Left Corner 3", "FGM")
    right_corner_fgm = _zone_numeric(current, "Right Corner 3", "FGM")
    corner_fga = left_corner_fga + right_corner_fga

    block = pd.DataFrame({"player_id": _numeric(current["PLAYER_ID"]).astype("Int64")})
    block["rim_attempt_rate"] = _safe_divide(restricted_fga, total_fga)
    block["paint_non_ra_attempt_rate"] = _safe_divide(paint_fga, total_fga)
    block["midrange_attempt_rate"] = _safe_divide(midrange_fga, total_fga)
    block["corner_three_attempt_rate"] = _safe_divide(corner_fga, total_fga)
    block["above_break_three_attempt_rate"] = _safe_divide(above_break_fga, total_fga)
    block["restricted_area_fg_pct"] = _zone_numeric(current, "Restricted Area", "FG_PCT")
    block["corner_three_fg_pct"] = _safe_divide(left_corner_fgm + right_corner_fgm, corner_fga)
    return block


def _build_tracking_features(player_features: pd.DataFrame, tracking_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    output = pd.DataFrame({"player_id": pd.to_numeric(player_features["player_id"], errors="coerce").astype("Int64")})
    if "minutes_total" in player_features.columns:
        minutes = _numeric(player_features["minutes_total"])
    else:
        minutes = pd.Series(0.0, index=player_features.index)
    output["_minutes_total"] = minutes
    if "field_goal_attempts" in player_features.columns:
        output["_field_goal_attempts"] = _numeric(player_features["field_goal_attempts"])
    else:
        output["_field_goal_attempts"] = 0.0

    for measure_type, frame in tracking_frames.items():
        if frame.empty:
            continue
        current = _flatten_columns(frame)
        validate_columns(current, ["PLAYER_ID"], dataset_name=f"{measure_type} tracking stats")
        current = current.rename(columns={"PLAYER_ID": "player_id"})
        measure_key = _normalize_key(measure_type)

        if measure_key == "catch_shoot":
            output = _merge_feature_block(output, _tracking_shot_block(current, "catch_shoot"))
        elif measure_key in {"pull_up", "pull_up_shot"}:
            output = _merge_feature_block(output, _tracking_shot_block(current, "pull_up"))
        elif measure_key == "drives":
            output = _merge_count_per_36(output, current, "drives_per_36", ["DRIVES"])
        elif measure_key == "paint_touches":
            output = _merge_count_per_36(output, current, "paint_touches_per_36", ["PAINT_TOUCHES", "PAINT_TOUCH"])
        elif measure_key == "post_touches":
            output = _merge_count_per_36(output, current, "post_touches_per_36", ["POST_TOUCHES", "POST_TOUCH"])
        elif measure_key == "elbow_touches":
            output = _merge_count_per_36(output, current, "elbow_touches_per_36", ["ELBOW_TOUCHES", "ELBOW_TOUCH"])
        elif measure_key in {"possessions", "passing"}:
            output = _merge_touch_context(output, current)

    if "catch_shoot_attempts" in output.columns:
        output["catch_shoot_attempt_frequency"] = _safe_divide(output["catch_shoot_attempts"], output["_field_goal_attempts"])
    if "pull_up_attempts" in output.columns:
        output["pull_up_attempt_frequency"] = _safe_divide(output["pull_up_attempts"], output["_field_goal_attempts"])
    if "passes_made" in output.columns and "touches" in output.columns:
        output["passes_made_per_touch"] = _safe_divide(output["passes_made"], output["touches"])

    return output.drop(
        columns=[
            "_minutes_total",
            "_field_goal_attempts",
            "catch_shoot_attempts",
            "pull_up_attempts",
            "passes_made",
            "touches",
        ],
        errors="ignore",
    )


def _tracking_shot_block(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    block = pd.DataFrame({"player_id": _numeric(frame["player_id"]).astype("Int64")})
    attempt_column = _find_first_column(frame, [f"{prefix}_FGA", f"{prefix}_SHOT_FGA", "FGA"])
    frequency_column = _find_first_column(frame, ["FGA_FREQUENCY"])
    efg_column = _find_first_column(frame, [f"{prefix}_EFG_PCT", "EFG_PCT"])

    if attempt_column is not None:
        block[f"{prefix}_attempts"] = _numeric(frame[attempt_column])
    if frequency_column is not None:
        block[f"{prefix}_attempt_frequency"] = _numeric(frame[frequency_column])
    if efg_column is not None:
        block[f"{prefix}_effective_fg_pct"] = _numeric(frame[efg_column])
    return block


def _merge_count_per_36(output: pd.DataFrame, frame: pd.DataFrame, feature_name: str, candidates: list[str]) -> pd.DataFrame:
    count_column = _find_first_column(frame, candidates)
    if count_column is None:
        return output

    block = pd.DataFrame(
        {
            "player_id": _numeric(frame["player_id"]).astype("Int64"),
            "_count": _numeric(frame[count_column]),
        }
    )
    merged = output.merge(block, on="player_id", how="left")
    merged[feature_name] = _safe_divide(merged["_count"] * 36.0, merged["_minutes_total"])
    return merged.drop(columns=["_count"])


def _merge_touch_context(output: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    block = pd.DataFrame({"player_id": _numeric(frame["player_id"]).astype("Int64")})
    frontcourt_column = _find_first_column(frame, ["FRONT_CT_TOUCHES", "FRONTCOURT_TOUCHES"])
    passes_column = _find_first_column(frame, ["PASSES_MADE", "PASSES"])
    touches_column = _find_first_column(frame, ["TOUCHES"])
    avg_seconds_column = _find_first_column(frame, ["AVG_SEC_PER_TOUCH", "AVG_SECONDS_PER_TOUCH"])

    if frontcourt_column is not None:
        block["frontcourt_touches_per_36"] = _numeric(frame[frontcourt_column])
    if passes_column is not None:
        block["passes_made"] = _numeric(frame[passes_column])
    if touches_column is not None:
        block["touches"] = _numeric(frame[touches_column])
    if passes_column is not None and touches_column is not None:
        block["passes_made_per_touch"] = _safe_divide(block["passes_made"], block["touches"])
    if avg_seconds_column is not None:
        block["avg_seconds_per_touch"] = _numeric(frame[avg_seconds_column])

    merged = _merge_feature_block(output, block)
    if "frontcourt_touches_per_36" in merged.columns:
        merged["frontcourt_touches_per_36"] = _safe_divide(merged["frontcourt_touches_per_36"] * 36.0, merged["_minutes_total"])
    return merged


def _merge_feature_block(output: pd.DataFrame, block: pd.DataFrame) -> pd.DataFrame:
    if block.empty:
        return output
    block = _aggregate_player_rows(block)
    merged = output.merge(block, on="player_id", how="left", suffixes=("", "_new"))
    for column in [column for column in merged.columns if column.endswith("_new")]:
        base_column = column.removesuffix("_new")
        merged[base_column] = merged[column].combine_first(merged.get(base_column))
        merged = merged.drop(columns=[column])
    return merged


def _combine_blocks(blocks: list[pd.DataFrame]) -> pd.DataFrame:
    if not blocks:
        return pd.DataFrame()
    combined = _aggregate_player_rows(blocks[0])
    for block in blocks[1:]:
        combined = _merge_feature_block(combined, block)
    return combined


def _aggregate_player_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse endpoint rows to one player row before joining to fingerprints."""
    if "player_id" not in frame.columns or not frame["player_id"].duplicated().any():
        return frame

    numeric_columns = [
        column
        for column in frame.columns
        if column != "player_id" and pd.api.types.is_numeric_dtype(frame[column])
    ]
    aggregated = frame.groupby("player_id", as_index=False)[numeric_columns].mean()
    return aggregated


def _source_player_ids(frames: object) -> set[int]:
    player_ids: set[int] = set()
    for frame in frames:
        if frame is None or frame.empty:
            continue
        current = _flatten_columns(frame)
        player_id_column = "PLAYER_ID" if "PLAYER_ID" in current.columns else "player_id"
        if player_id_column not in current.columns:
            continue
        ids = pd.to_numeric(current[player_id_column], errors="coerce").dropna().astype(int)
        player_ids.update(ids.tolist())
    return player_ids


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    current = frame.copy()
    current.columns = [
        " ".join(str(part) for part in column if str(part) and str(part) != "nan")
        if isinstance(column, tuple)
        else str(column)
        for column in current.columns
    ]
    return current


def _zone_numeric(frame: pd.DataFrame, zone: str, stat: str) -> pd.Series:
    column = _find_zone_column(frame, zone, stat)
    if column is None:
        return pd.Series(0.0, index=frame.index)
    return _numeric(frame[column])


def _find_zone_column(frame: pd.DataFrame, zone: str, stat: str) -> str | None:
    zone_tokens = _tokens(zone)
    stat_tokens = _tokens(stat)
    for column in frame.columns:
        column_tokens = _tokens(column)
        if all(token in column_tokens for token in zone_tokens) and all(token in column_tokens for token in stat_tokens):
            return column
    return None


def _find_first_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized_columns = {_normalize_column(column): column for column in frame.columns}
    for candidate in candidates:
        normalized = _normalize_column(candidate)
        if normalized in normalized_columns:
            return normalized_columns[normalized]
    return None


def _optional_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return _numeric(frame[column])


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.astype(float).divide(denominator.astype(float).replace(0, np.nan))
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _normalize_key(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _normalize_column(value: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


def _tokens(value: object) -> set[str]:
    return {token for token in re.split(r"[^A-Z0-9]+", str(value).upper()) if token}

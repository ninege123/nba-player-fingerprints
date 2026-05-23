"""Local cache helpers for API data pulls."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW_DATA_DIR = Path("data/raw")


def cache_path(
    dataset_name: str,
    season: str,
    cache_dir: Path | str = RAW_DATA_DIR,
    file_format: str = "parquet",
) -> Path:
    """Build a stable cache path for a season-level dataset."""
    if file_format not in {"parquet", "csv"}:
        raise ValueError("file_format must be 'parquet' or 'csv'")

    safe_season = season.replace("-", "_")
    return Path(cache_dir) / f"{dataset_name}_{safe_season}.{file_format}"


def write_cached_frame(frame: pd.DataFrame, path: Path | str) -> Path:
    """Write a DataFrame cache file and return its path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".parquet":
        frame.to_parquet(output_path, index=False)
    elif output_path.suffix == ".csv":
        frame.to_csv(output_path, index=False)
    else:
        raise ValueError("Cache path must end in .parquet or .csv")

    return output_path


def read_cached_frame(path: Path | str) -> pd.DataFrame:
    """Read a cached DataFrame from parquet or CSV."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if input_path.suffix == ".parquet":
        return pd.read_parquet(input_path)
    if input_path.suffix == ".csv":
        return pd.read_csv(input_path)

    raise ValueError("Cache path must end in .parquet or .csv")


def cached_frame_exists(path: Path | str) -> bool:
    """Return whether a cache file exists."""
    return Path(path).exists()

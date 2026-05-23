# Data Sources

## Initial source

- `nba_api`: Python client for public NBA Stats endpoints.

## Planned tables

- Player season traditional stats from `LeagueDashPlayerStats` with `measure_type="Base"` and `per_mode="Totals"`.
- Player season advanced stats from the same endpoint with `measure_type="Advanced"`.
- Player index metadata from `PlayerIndex` for NBA-listed broad position labels.

## Later sources

- Play type data if a stable public source is available.
- Shot chart or location data for richer shooting fingerprints.
- Tracking-derived style features if available.

## Storage rules

- Raw API responses belong in `data/raw/`.
- Processed feature tables belong in `data/processed/`.
- Large raw and processed files are gitignored.
- Small toy data for tests can live in `data/sample/`.

## Current ingestion entry point

Use `nba_fingerprints.data.nba_api_client.load_player_season_stats()` to load one player-season stat table. The function checks the local cache first and only calls the NBA Stats API when a cache file is absent or cache use is disabled.

Use `nba_fingerprints.data.nba_api_client.load_player_index()` to load player metadata, including `POSITION`. In the current NBA endpoint this is a broad listed label such as `G`, `F`, `C`, `G-F`, or `F-C`, not a guaranteed `PG/SG/SF/PF/C` role label.

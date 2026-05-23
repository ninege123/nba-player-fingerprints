# Data Sources

## Initial source

- `nba_api`: Python client for public NBA Stats endpoints.

## Planned tables

- Player season traditional stats.
- Player season advanced stats.
- Player biographical or common info for position labels.

## Later sources

- Play type data if a stable public source is available.
- Shot chart or location data for richer shooting fingerprints.
- Tracking-derived style features if available.

## Storage rules

- Raw API responses belong in `data/raw/`.
- Processed feature tables belong in `data/processed/`.
- Large raw and processed files are gitignored.
- Small toy data for tests can live in `data/sample/`.

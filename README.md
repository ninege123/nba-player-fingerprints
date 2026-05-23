# NBA Player Fingerprints

This project builds player-season fingerprints for NBA players using public basketball data. A fingerprint is a normalized feature vector that describes a player's role, position tendencies, and style of play.

The goal is to compare players to position references, style archetypes, and other players using similarity analysis, outlier detection, and explainable visualizations.

## Project Motivation

Traditional NBA analysis often focuses on box-score totals or prediction tasks. This project focuses on player identity: what does a player's statistical profile look like, and which players or archetypes are most similar?

The methodology is inspired by my professional fingerprinting workflows at work used to compare behavioral patterns across peer groups.

## Planned Methodology

1. Pull public NBA player-season data.
2. Build explicit player-level fingerprint features.
3. Normalize features for profile comparison.
4. Create reference fingerprints by position and manual archetype.
5. Calculate cosine similarity between players and reference groups.
6. Identify hybrid players, specialists, and outliers.
7. Visualize player similarity and style clusters.

## Repository Structure

- `src/nba_fingerprints/`: reusable Python package code.
- `docs/`: project design, feature definitions, and data-source notes.
- `notebooks/`: exploratory analysis and portfolio storytelling.
- `R/`: optional R exploration, modeling, and visualization.
- `data/raw/`: raw API outputs, gitignored.
- `data/processed/`: processed feature tables, gitignored.
- `data/sample/`: small test/sample files if needed.
- `reports/`: generated analysis outputs and writeups.
- `tests/`: automated checks for reusable Python logic.

## Tools

- Python
- R
- pandas
- scikit-learn
- nba_api
- DuckDB, if useful later
- Git/GitHub

## Development

Run the current Python checks with:

```powershell
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m unittest discover
```

Load one season of player stats from Python:

```python
from nba_fingerprints.data.nba_api_client import load_player_season_stats
from nba_fingerprints.features.fingerprints import build_fingerprint_table
from nba_fingerprints.features.player_season import build_player_season_features
from nba_fingerprints.models.neighbors import find_nearest_neighbors

players = load_player_season_stats("2023-24")
features = build_player_season_features(players, season="2023-24", min_minutes=500)
fingerprints = build_fingerprint_table(features)
neighbors = find_nearest_neighbors(fingerprints, top_n=5)

print(neighbors.head())
```

Export processed CSVs for manual inspection:

```powershell
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m nba_fingerprints.cli export-player-season --season 2023-24 --min-minutes 500 --top-n 5
```

This writes:

- `data/processed/player_season_features_2023_24.csv`
- `data/processed/player_fingerprints_2023_24.csv`
- `data/processed/player_neighbors_2023_24.csv`
- `data/processed/position_references_2023_24.csv`
- `data/processed/player_position_scores_2023_24.csv`
- `data/processed/archetype_references_2023_24.csv`
- `data/processed/player_archetype_scores_2023_24.csv`
- `data/processed/player_archetype_explanations_2023_24.csv`

Render the portfolio report with Quarto:

```powershell
quarto render reports/player_fingerprint_report.qmd
```

Publish the MVP report through GitHub Pages:

```powershell
Copy-Item reports/player_fingerprint_report.html site/index.html -Force
Copy-Item reports/player_fingerprint_report_files site/player_fingerprint_report_files -Recurse -Force
git add site .github/workflows/pages.yml
git commit -m "Publish MVP report site"
git push origin main
```

The GitHub Pages workflow publishes the committed `site/` folder as a static site.


# NBA Player Fingerprints

This project builds player-season fingerprints for NBA players using public basketball data. A fingerprint is a normalized feature vector that describes a player's role, position tendencies, and style of play using box-score and advanced-stat indicators.

The goal is to compare players to position references, style archetypes, and other players using similarity analysis, outlier detection, and explainable visualizations.

## Project Motivation

Traditional NBA analysis often focuses on box-score totals or prediction tasks. This project focuses on player identity: what does a player's statistical profile look like, and which players or archetypes are most similar?

The methodology is inspired by my professional fingerprinting workflows at work used to compare behavioral patterns across peer groups.

## Methodology

This project treats each player-season as a statistical profile rather than a single ranking or prediction target. The core unit of analysis is:

```text
one row = one NBA player in one season
```

The workflow converts public NBA box-score and advanced-stat data into normalized feature vectors, then compares those vectors across players, listed positions, and style archetypes.

### Feature Engineering

Raw counting stats are transformed into rate-based features so players with different minutes and roles can be compared more fairly. The current feature set includes:

- per-36 production: scoring, assists, rebounds, steals, blocks, turnovers, and fouls
- shooting profile: three-point attempt rate, free-throw attempt rate, effective field-goal percentage, true shooting percentage
- role and creation: usage rate, assist percentage, assist-to-turnover ratio
- rebounding profile: offensive, defensive, and total rebound percentages
- context and impact: pace, net rating, player impact estimate

This avoids comparing players only by raw totals, which would mostly measure playing time and team context.

### Normalization

Features live on different scales: true shooting is a percentage, pace is a possession estimate, per-36 stats are volume rates, and net rating can be negative. Before similarity scoring, features are min-max scaled to a common 0-to-1 range within the comparison set.

This makes the fingerprint represent the shape of a player's statistical profile, not the original units of the columns. The tradeoff is that min-max scaling is sensitive to extreme values, so future versions may compare robust scaling, percentile ranks, and z-score normalization.

### Similarity Analysis

The project uses cosine similarity to compare fingerprint vectors. Cosine similarity is useful here because it measures profile direction: two players can be considered stylistically similar when their normalized feature patterns point in the same direction, even if their exact magnitudes differ.

Similarity is calculated for:

- player-to-player nearest neighbors
- player-to-listed-position references
- player-to-manual-archetype references

Cosine similarity is not treated as a causal metric or a player quality score. It is a descriptive measure of statistical profile resemblance.

### Reference Fingerprints

Reference fingerprints are built in two ways:

- listed-position references: average fingerprints for NBA-listed broad positions such as `G`, `F`, and `C`
- manual archetype references: transparent weighted profiles for basketball roles such as `floor_general`, `scoring_guard`, `three_and_d_wing`, `stretch_big`, and `rim_running_center`

Position labels are used as loose anchors rather than ground truth. NBA public metadata is broad and imperfect, so the archetype layer is where the project expresses richer basketball role language.

### Sample Size Controls

The default export uses a minimum-minutes threshold of 500 season minutes. This reduces noise from small samples, short stints, and low-minute players whose rate stats can be unstable. The raw API pull is still cached locally, so lower thresholds can be used for exploratory review.

### Explainability

For each player's top archetype match, the pipeline produces:

- supporting features: dimensions where the player strongly overlaps the archetype reference
- gap features: dimensions where the player differs most from the archetype reference

This makes the output more interpretable than a similarity score alone. A result should be explainable in basketball language, not only by a numeric score.

### Future Statistical Extensions

Planned extensions include:

- outlier detection using distance from position or archetype references
- hybrid labeling based on close scores across multiple archetypes
- Jensen-Shannon divergence for distribution-based comparisons
- population stability index to compare player profiles across seasons
- confidence flags based on minutes, games played, and feature volatility
- clustering to compare manual archetypes against data-derived style groups

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
- `data/processed/player_summary_2023_24.csv`

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


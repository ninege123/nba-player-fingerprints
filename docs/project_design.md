# Project Design

## Goal

Build player-season fingerprints for NBA players using public basketball data. A fingerprint is a normalized feature vector that describes how a player contributes across scoring, shooting, playmaking, rebounding, defense, and role/context.

## First version scope

- Grain: one row per player per season.
- Data: public NBA player season box-score and advanced stat tables.
- Implementation: Python package code for ingestion, feature engineering, and similarity; notebooks for exploration and portfolio storytelling.
- Comparisons: player-to-listed-position, player-to-archetype, and player-to-player similarity.

## Pipeline

1. Ingest public NBA player-season data.
2. Cache raw API responses under `data/raw/`.
3. Clean and standardize identifiers, season labels, positions, and numeric stats.
4. Engineer explicit fingerprint features from raw stats.
5. Normalize fingerprint features so players can be compared on profile shape.
6. Compute cosine similarity and nearest neighbors.
7. Build reference fingerprints by position and manually defined archetype.
8. Explain matches with feature-level differences and basketball-language summaries.

## Initial archetypes

- Floor general
- Scoring guard
- 3-and-D wing
- Point forward
- Stretch big
- Rim-running center
- Interior defensive big
- High-usage creator

Manual archetypes are the first pass because they are easier to explain. Clustering can be added after the pipeline is stable.

## Validation

- Check raw and processed schemas before scoring.
- Require minimum minutes or games played for reference fingerprints.
- Compare obvious player examples against basketball intuition.
- Track sensitivity to normalization choices.

## Current implementation status

- Player-season box-score totals can be loaded through `nba_api` and cached locally.
- First-pass player-season features can be engineered from box-score totals.
- Features can be min-max scaled into normalized fingerprint tables.
- Player-to-player nearest neighbors can be scored with cosine similarity.
- Processed feature, fingerprint, and neighbor tables can be exported to `data/processed/`.
- NBA-listed broad position references (`G`, `F`, `C`) can be built and scored. Granular role labels such as floor general, scoring guard, 3-and-D wing, and stretch big should come from the archetype layer rather than treating listed positions as absolute truth.
- Manual archetype references can be built from transparent feature weights and scored against every player fingerprint.
- Top archetype matches can be explained with supporting feature dimensions and largest gaps.
- A Quarto report can read the processed outputs and present portfolio-facing tables and base R charts.

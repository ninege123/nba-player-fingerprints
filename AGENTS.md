# AGENTS.md

## Project goal

This project builds NBA player fingerprints: feature vectors that describe players by role, position, and style of play. The project should support similarity analysis, position archetypes, outlier detection, and explainable visualizations.

## Preferred workflow

- Before making large changes, inspect the repo and propose a plan.
- Prefer small, reviewable changes over big rewrites.
- Keep exploratory notebooks/scripts separate from reusable source code.
- After changing code, run the relevant checks and summarize what passed or failed.
- Do not commit secrets, API keys, large raw data files, or private/company code.

## Stack preference

- R is acceptable and preferred for exploratory analysis, modeling, and visualization.
- Python is acceptable for data ingestion, package APIs, dashboards, or deployment if it makes the project easier.
- SQL is acceptable for transformations and similarity calculations.
- Keep the project reproducible.

## Data rules

- Do not commit large raw datasets.
- Put raw data under `data/raw/` and keep it gitignored.
- Put small toy/sample files under `data/sample/` only if useful for tests.
- Document data sources in `docs/data_sources.md`.

## Suggested structure

- `README.md`: project overview and how to run
- `docs/`: methodology and data-source notes
- `sql/`: SQL transformations
- `R/` or `src/`: reusable functions
- `notebooks/` or `analysis/`: exploratory work
- `data/sample/`: small non-sensitive sample data
- `outputs/`: generated charts/tables, gitignored if large

## Code style

- Use clear, descriptive names.
- Prefer `snake_case`.
- Add comments for non-obvious statistical or basketball logic.
- Keep functions focused and testable.

## Basketball methodology rules

- Separate raw box-score/stat data from engineered fingerprint features.
- Make feature definitions explicit.
- Avoid treating position labels as absolute truth; use them as reference anchors.
- When computing similarity, document normalization choices.
- Explain results in basketball language, not only math terms.

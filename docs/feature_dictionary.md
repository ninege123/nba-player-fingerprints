# Feature Dictionary

This file defines the first-pass player-season fingerprint dimensions. The exact column names may change once the ingestion source is finalized, but every engineered feature should keep a clear basketball interpretation.

## Role and volume

- `minutes_per_game`: How large the player's role is.
- `usage_rate`: Share of team possessions used while on the floor.
- `points_per_36`: Scoring volume normalized to 36 minutes.

## Scoring and shooting

- `three_point_attempt_rate`: Share of field-goal attempts from three.
- `free_throw_attempt_rate`: Free-throw attempts per field-goal attempt.
- `true_shooting_pct`: Overall scoring efficiency.
- `effective_fg_pct`: Shooting efficiency adjusted for three-pointers.

## Playmaking

- `assist_rate`: Share of teammate field goals assisted while on the floor.
- `turnover_rate`: Turnovers per possession estimate.
- `assist_to_turnover`: Passing creation relative to mistakes.

## Rebounding

- `offensive_rebound_rate`: Offensive rebounding involvement.
- `defensive_rebound_rate`: Defensive rebounding involvement.
- `total_rebound_rate`: Overall rebounding involvement.

## Defense proxies

- `steal_rate`: Ball pressure and event creation proxy.
- `block_rate`: Rim protection and shot-deterrence proxy.
- `personal_fouls_per_36`: Defensive physicality or discipline proxy.

## Normalization

The first implementation uses min-max scaling for interpretable 0-to-1 feature vectors. Future versions may compare z-score scaling, robust scaling, percentile ranks, PSI, and Jensen-Shannon divergence.

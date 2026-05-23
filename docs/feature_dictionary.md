# Feature Dictionary

This file defines the first-pass player-season fingerprint dimensions. The exact column names may change once the ingestion source is finalized, but every engineered feature should keep a clear basketball interpretation.

## Role and volume

- `minutes_per_game`: How large the player's role is.
- `points_per_36`: Scoring volume normalized to 36 minutes.

## Scoring and shooting

- `three_point_attempt_rate`: Share of field-goal attempts from three.
- `free_throw_attempt_rate`: Free-throw attempts per field-goal attempt.
- `true_shooting_pct`: Overall scoring efficiency.
- `effective_fg_pct`: Shooting efficiency adjusted for three-pointers.

## Playmaking

- `assists_per_36`: Assist volume normalized to 36 minutes.
- `turnovers_per_36`: Turnover volume normalized to 36 minutes.
- `assist_to_turnover`: Passing creation relative to mistakes.

## Rebounding

- `offensive_rebounds_per_36`: Offensive rebounding involvement.
- `defensive_rebounds_per_36`: Defensive rebounding involvement.
- `rebounds_per_36`: Overall rebounding involvement.

## Defense proxies

- `steals_per_36`: Ball pressure and event creation proxy.
- `blocks_per_36`: Rim protection and shot-deterrence proxy.
- `personal_fouls_per_36`: Defensive physicality or discipline proxy.

## Later feature candidates

- `usage_rate`: Share of team possessions used while on the floor.
- `assist_rate`: Share of teammate field goals assisted while on the floor.
- `turnover_rate`: Turnovers per possession estimate.
- Position labels and play-type frequencies once reliable public sources are integrated.

## Normalization

The first implementation uses min-max scaling for interpretable 0-to-1 feature vectors. Cosine similarity is then computed on the normalized feature vectors. Future versions may compare z-score scaling, robust scaling, percentile ranks, PSI, and Jensen-Shannon divergence.

# Feature Dictionary

This file defines the first-pass player-season fingerprint dimensions. The exact column names may change once the ingestion source is finalized, but every engineered feature should keep a clear basketball interpretation.

## Role and volume

- `minutes_per_game`: How large the player's role is.
- `points_per_36`: Scoring volume normalized to 36 minutes.
- `usage_rate`: Share of team possessions used while the player is on the floor.
- `pace`: Possession environment while the player is on the floor.
- `player_impact_estimate`: NBA advanced-stat estimate of all-around impact.
- `net_rating`: Team point differential per 100 possessions while the player is on the floor.

## Scoring and shooting

- `three_point_attempt_rate`: Share of field-goal attempts from three.
- `free_throw_attempt_rate`: Free-throw attempts per field-goal attempt.
- `true_shooting_pct`: Overall scoring efficiency.
- `effective_fg_pct`: Shooting efficiency adjusted for three-pointers.

## Playmaking

- `assists_per_36`: Assist volume normalized to 36 minutes.
- `turnovers_per_36`: Turnover volume normalized to 36 minutes.
- `assist_to_turnover`: Passing creation relative to mistakes.
- `assist_pct`: Share of teammate field goals assisted while the player is on the floor.
- `turnover_pct`: Team turnover percentage while the player is on the floor.

## Rebounding

- `offensive_rebounds_per_36`: Offensive rebounding involvement.
- `defensive_rebounds_per_36`: Defensive rebounding involvement.
- `rebounds_per_36`: Overall rebounding involvement.
- `offensive_rebound_pct`: Offensive rebound percentage while on the floor.
- `defensive_rebound_pct`: Defensive rebound percentage while on the floor.
- `total_rebound_pct`: Total rebound percentage while on the floor.

## Defense proxies

- `steals_per_36`: Ball pressure and event creation proxy.
- `blocks_per_36`: Rim protection and shot-deterrence proxy.
- `personal_fouls_per_36`: Defensive physicality or discipline proxy.

## Later feature candidates

- Position labels and play-type frequencies once reliable public sources are integrated.

## Manual archetype references

The first archetype layer uses transparent feature weights over the normalized fingerprint columns. The current weights combine box-score rates with NBA advanced indicators such as `usage_rate`, `assist_pct`, rebound percentages, and `player_impact_estimate`. These references are intentionally simple and explainable:

- `floor_general`
- `scoring_guard`
- `three_and_d_wing`
- `point_forward`
- `stretch_big`
- `rim_running_center`
- `interior_defensive_big`
- `high_usage_creator`

The `player_archetype_explanations` output summarizes each player's top-ranked archetype match with the strongest supporting normalized feature dimensions and the largest gaps from the archetype reference.

## Normalization

The first implementation uses min-max scaling for interpretable 0-to-1 feature vectors. Cosine similarity is then computed on the normalized feature vectors. Future versions may compare z-score scaling, robust scaling, percentile ranks, PSI, and Jensen-Shannon divergence.

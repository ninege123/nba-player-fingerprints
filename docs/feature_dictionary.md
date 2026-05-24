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

## Scoring style

These dimensions describe how a player scores, not only how often or how efficiently.
They are optional because they require NBA Stats shot-location, tracking, or play-type
tables beyond the base box-score and advanced-stat endpoints.

### Play type

Play-type features come from offensive Synergy play-type summaries. Frequency columns
describe how often the player's possessions ended in that action type; efficiency columns
describe the outcome of those possessions.

- `isolation_frequency`: Share of possessions finished from isolation.
- `isolation_points_per_possession`: Isolation scoring efficiency.
- `isolation_score_frequency`: Share of isolation possessions that scored.
- `isolation_turnover_frequency`: Share of isolation possessions ending in turnovers.
- `pnr_ball_handler_frequency`: Pick-and-roll ball-handler possession share.
- `pnr_ball_handler_points_per_possession`: Pick-and-roll ball-handler efficiency.
- `pnr_roll_man_frequency`: Pick-and-roll roll-man possession share.
- `pnr_roll_man_points_per_possession`: Pick-and-roll roll-man efficiency.
- `spot_up_frequency`: Spot-up possession share.
- `spot_up_points_per_possession`: Spot-up scoring efficiency.
- `cut_frequency`: Cut possession share.
- `cut_points_per_possession`: Cut scoring efficiency.
- `handoff_frequency`: Handoff possession share.
- `handoff_points_per_possession`: Handoff scoring efficiency.
- `post_up_frequency`: Post-up possession share.
- `post_up_points_per_possession`: Post-up scoring efficiency.
- `off_screen_frequency`: Off-screen possession share.
- `off_screen_points_per_possession`: Off-screen scoring efficiency.
- `putback_frequency`: Putback possession share.
- `putback_points_per_possession`: Putback scoring efficiency.
- `transition_frequency`: Transition possession share.
- `transition_points_per_possession`: Transition scoring efficiency.

### Shot locations

- `rim_attempt_rate`: Share of attempts from the restricted area.
- `paint_non_ra_attempt_rate`: Share of attempts from the paint outside the restricted area.
- `midrange_attempt_rate`: Share of attempts from mid-range.
- `corner_three_attempt_rate`: Share of attempts from either corner three zone.
- `above_break_three_attempt_rate`: Share of attempts from above the break three.
- `restricted_area_fg_pct`: Field-goal percentage at the rim.
- `corner_three_fg_pct`: Field-goal percentage on corner threes.

### Tracking shots and touches

- `catch_shoot_attempt_frequency`: Share of attempts classified as catch-and-shoot.
- `catch_shoot_effective_fg_pct`: Effective field-goal percentage on catch-and-shoot attempts.
- `pull_up_attempt_frequency`: Share of attempts classified as pull-up attempts.
- `pull_up_effective_fg_pct`: Effective field-goal percentage on pull-up attempts.
- `drives_per_36`: Drives normalized to 36 minutes.
- `paint_touches_per_36`: Paint touches normalized to 36 minutes.
- `post_touches_per_36`: Post touches normalized to 36 minutes.
- `elbow_touches_per_36`: Elbow touches normalized to 36 minutes.
- `frontcourt_touches_per_36`: Frontcourt touches normalized to 36 minutes.
- `passes_made_per_touch`: Passing activity relative to touches.
- `avg_seconds_per_touch`: Average time the player holds the ball per touch.

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

- Defensive shot contests and matchup data once reliable public sources are integrated.

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
- `isolation_creator`
- `pick_and_roll_creator`
- `roll_finisher`
- `movement_shooter`
- `spot_up_spacer`
- `cutter_slasher`
- `post_scorer`
- `transition_attacker`

The `player_archetype_explanations` output summarizes each player's top-ranked archetype match with the strongest supporting normalized feature dimensions and the largest gaps from the archetype reference.

## Normalization

The first implementation uses min-max scaling for interpretable 0-to-1 feature vectors. Cosine similarity is then computed on the normalized feature vectors. Future versions may compare z-score scaling, robust scaling, percentile ranks, PSI, and Jensen-Shannon divergence.

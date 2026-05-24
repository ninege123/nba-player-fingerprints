import unittest

import pandas as pd

from nba_fingerprints.features.scoring_style import (
    SCORING_STYLE_AVAILABILITY_COLUMNS,
    SCORING_STYLE_FEATURE_COLUMNS,
    attach_scoring_style_features,
    build_scoring_style_features,
)


class ScoringStyleFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.player_features = pd.DataFrame(
            {
                "player_id": [1, 2],
                "player_name": ["Creator", "Spacer"],
                "minutes_total": [720, 360],
            }
        )

    def test_build_scoring_style_features_defaults_to_zero(self) -> None:
        features = build_scoring_style_features(self.player_features)

        self.assertEqual(features.columns.tolist(), ["player_id", *SCORING_STYLE_FEATURE_COLUMNS, *SCORING_STYLE_AVAILABILITY_COLUMNS])
        self.assertTrue((features[SCORING_STYLE_FEATURE_COLUMNS] == 0.0).all(axis=None))
        self.assertTrue((features[SCORING_STYLE_AVAILABILITY_COLUMNS] == False).all(axis=None))

    def test_build_scoring_style_features_uses_synergy_play_type_stats(self) -> None:
        synergy = {
            "isolation": pd.DataFrame(
                {
                    "PLAYER_ID": [1],
                    "POSS_PCT": [0.18],
                    "PPP": [1.12],
                    "SCORE_POSS_PCT": [0.48],
                    "TOV_POSS_PCT": [0.08],
                    "EFG_PCT": [0.54],
                    "PERCENTILE": [0.87],
                }
            )
        }

        features = build_scoring_style_features(self.player_features, synergy_frames=synergy)
        row = features[features["player_id"] == 1].iloc[0]

        self.assertEqual(row["isolation_frequency"], 0.18)
        self.assertEqual(row["isolation_points_per_possession"], 1.12)
        self.assertEqual(row["isolation_score_frequency"], 0.48)
        self.assertEqual(row["isolation_turnover_frequency"], 0.08)
        self.assertEqual(row["isolation_effective_fg_pct"], 0.54)
        self.assertEqual(row["isolation_percentile"], 0.87)
        self.assertTrue(row["has_synergy_data"])

    def test_build_scoring_style_features_uses_shot_locations(self) -> None:
        shot_locations = pd.DataFrame(
            {
                "PLAYER_ID": [1],
                "Restricted Area FGM": [50],
                "Restricted Area FGA": [80],
                "Restricted Area FG_PCT": [0.625],
                "In The Paint (Non-RA) FGA": [20],
                "Mid-Range FGA": [30],
                "Left Corner 3 FGM": [6],
                "Left Corner 3 FGA": [15],
                "Right Corner 3 FGM": [4],
                "Right Corner 3 FGA": [10],
                "Above the Break 3 FGA": [45],
                "Backcourt FGA": [0],
            }
        )

        features = build_scoring_style_features(self.player_features, shot_locations=shot_locations)
        row = features[features["player_id"] == 1].iloc[0]

        self.assertEqual(row["rim_attempt_rate"], 0.4)
        self.assertEqual(row["paint_non_ra_attempt_rate"], 0.1)
        self.assertEqual(row["midrange_attempt_rate"], 0.15)
        self.assertEqual(row["corner_three_attempt_rate"], 0.125)
        self.assertEqual(row["above_break_three_attempt_rate"], 0.225)
        self.assertEqual(row["restricted_area_fg_pct"], 0.625)
        self.assertEqual(row["corner_three_fg_pct"], 0.4)
        self.assertTrue(row["has_shot_location_data"])

    def test_build_scoring_style_features_uses_tracking_stats(self) -> None:
        tracking = {
            "catch_shoot": pd.DataFrame({"PLAYER_ID": [1], "FGA_FREQUENCY": [0.30], "EFG_PCT": [0.62]}),
            "pull_up": pd.DataFrame({"PLAYER_ID": [1], "FGA_FREQUENCY": [0.22], "EFG_PCT": [0.50]}),
            "drives": pd.DataFrame({"PLAYER_ID": [1], "DRIVES": [200]}),
            "paint_touches": pd.DataFrame({"PLAYER_ID": [1], "PAINT_TOUCHES": [100]}),
            "post_touches": pd.DataFrame({"PLAYER_ID": [1], "POST_TOUCHES": [40]}),
            "elbow_touches": pd.DataFrame({"PLAYER_ID": [1], "ELBOW_TOUCHES": [20]}),
            "passing": pd.DataFrame(
                {
                    "PLAYER_ID": [1],
                    "FRONT_CT_TOUCHES": [500],
                    "PASSES_MADE": [400],
                    "TOUCHES": [800],
                    "AVG_SEC_PER_TOUCH": [4.2],
                }
            ),
        }

        features = build_scoring_style_features(self.player_features, tracking_frames=tracking)
        row = features[features["player_id"] == 1].iloc[0]

        self.assertEqual(row["catch_shoot_attempt_frequency"], 0.30)
        self.assertEqual(row["catch_shoot_effective_fg_pct"], 0.62)
        self.assertEqual(row["pull_up_attempt_frequency"], 0.22)
        self.assertEqual(row["pull_up_effective_fg_pct"], 0.50)
        self.assertEqual(row["drives_per_36"], 10.0)
        self.assertEqual(row["paint_touches_per_36"], 5.0)
        self.assertEqual(row["post_touches_per_36"], 2.0)
        self.assertEqual(row["elbow_touches_per_36"], 1.0)
        self.assertEqual(row["frontcourt_touches_per_36"], 25.0)
        self.assertEqual(row["passes_made_per_touch"], 0.5)
        self.assertEqual(row["avg_seconds_per_touch"], 4.2)
        self.assertTrue(row["has_tracking_shot_data"])
        self.assertTrue(row["has_touch_tracking_data"])

    def test_attach_scoring_style_features_keeps_existing_columns(self) -> None:
        attached = attach_scoring_style_features(
            self.player_features,
            synergy_frames={"spot_up": pd.DataFrame({"PLAYER_ID": [2], "POSS_PCT": [0.25], "PPP": [1.2]})},
        )

        self.assertIn("player_name", attached.columns)
        self.assertEqual(attached.loc[attached["player_id"] == 2, "spot_up_frequency"].iloc[0], 0.25)
        self.assertFalse(attached.loc[attached["player_id"] == 1, "has_synergy_data"].iloc[0])
        self.assertTrue(attached.loc[attached["player_id"] == 2, "has_synergy_data"].iloc[0])


if __name__ == "__main__":
    unittest.main()

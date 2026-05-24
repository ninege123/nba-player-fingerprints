import unittest

import pandas as pd

from nba_fingerprints.app.contracts import (
    build_app_feature_metadata,
    build_app_player_profiles,
    build_app_similarity_edges,
)


class AppContractTests(unittest.TestCase):
    def test_build_app_player_profiles_combines_summary_and_style_features(self) -> None:
        features = pd.DataFrame(
            {
                "season": ["2023-24"],
                "player_id": [1],
                "player_name": ["Creator"],
                "team_abbreviation": ["AAA"],
                "isolation_frequency": [0.20],
                "isolation_points_per_possession": [1.10],
                "pnr_ball_handler_frequency": [0.30],
                "pnr_ball_handler_points_per_possession": [0.95],
                "pnr_roll_man_frequency": [0.0],
                "pnr_roll_man_points_per_possession": [0.0],
                "spot_up_frequency": [0.05],
                "spot_up_points_per_possession": [1.0],
                "cut_frequency": [0.01],
                "cut_points_per_possession": [1.2],
                "handoff_frequency": [0.02],
                "handoff_points_per_possession": [1.0],
                "post_up_frequency": [0.0],
                "post_up_points_per_possession": [0.0],
                "off_screen_frequency": [0.03],
                "off_screen_points_per_possession": [1.1],
                "transition_frequency": [0.10],
                "transition_points_per_possession": [1.2],
                "rim_attempt_rate": [0.25],
                "midrange_attempt_rate": [0.15],
                "corner_three_attempt_rate": [0.05],
                "above_break_three_attempt_rate": [0.30],
                "catch_shoot_attempt_frequency": [0.20],
                "catch_shoot_effective_fg_pct": [0.60],
                "pull_up_attempt_frequency": [0.35],
                "pull_up_effective_fg_pct": [0.50],
                "drives_per_36": [12.0],
                "frontcourt_touches_per_36": [40.0],
                "passes_made_per_touch": [0.5],
                "avg_seconds_per_touch": [4.0],
                "has_synergy_data": [True],
                "has_shot_location_data": [True],
                "has_tracking_shot_data": [True],
                "has_touch_tracking_data": [True],
            }
        )
        summary = pd.DataFrame(
            {
                "season": ["2023-24"],
                "player_id": [1],
                "player_name": ["Creator"],
                "team_abbreviation": ["AAA"],
                "position": ["G"],
                "primary_position": ["G"],
                "games_played": [70],
                "minutes_total": [2000],
                "minutes_per_game": [28.6],
                "points_per_36": [24.0],
                "usage_rate": [0.28],
                "true_shooting_pct": [0.60],
                "top_neighbor": ["Similar Guard"],
                "top_neighbor_similarity": [0.92],
                "top_archetype": ["pick_and_roll_creator"],
                "top_archetype_similarity": [0.88],
                "top_position_reference": ["G"],
                "top_position_similarity": [0.86],
                "supporting_features": ["pnr_ball_handler_frequency=0.800"],
                "gap_features": ["assist_pct=0.100"],
            }
        )

        profiles = build_app_player_profiles(features, summary)

        self.assertEqual(profiles.shape[0], 1)
        self.assertEqual(profiles.loc[0, "top_archetype"], "pick_and_roll_creator")
        self.assertIn("pick-and-roll ball handler", profiles.loc[0, "style_summary"])
        self.assertTrue(profiles.loc[0, "has_synergy_data"])
        self.assertEqual(profiles.loc[0, "peer_typicality_percentile"], 1.0)
        self.assertAlmostEqual(profiles.loc[0, "distinctiveness_score"], 0.08)

    def test_build_app_similarity_edges_keeps_neighbor_columns(self) -> None:
        neighbors = pd.DataFrame(
            {
                "season": ["2023-24"],
                "player_id": [1],
                "player_name": ["Creator"],
                "team_abbreviation": ["AAA"],
                "neighbor_rank": [1],
                "neighbor_player_id": [2],
                "neighbor_player_name": ["Similar Guard"],
                "neighbor_team_abbreviation": ["BBB"],
                "cosine_similarity": [0.91],
            }
        )

        edges = build_app_similarity_edges(neighbors)

        self.assertEqual(edges.loc[0, "neighbor_player_name"], "Similar Guard")
        self.assertEqual(edges.loc[0, "cosine_similarity"], 0.91)

    def test_build_app_feature_metadata_marks_availability_flags_as_not_fingerprint_dimensions(self) -> None:
        metadata = build_app_feature_metadata(feature_columns=["points_per_36", "isolation_frequency"])

        self.assertEqual(metadata.loc[metadata["feature"] == "points_per_36", "group"].iloc[0], "scoring")
        self.assertEqual(metadata.loc[metadata["feature"] == "isolation_frequency", "group"].iloc[0], "play_type")
        self.assertFalse(metadata.loc[metadata["feature"] == "has_synergy_data", "is_fingerprint_dimension"].iloc[0])


if __name__ == "__main__":
    unittest.main()

import unittest

import pandas as pd

from nba_fingerprints.features.player_season import FINGERPRINT_FEATURE_COLUMNS, build_player_season_features


class PlayerSeasonFeatureTests(unittest.TestCase):
    def test_build_player_season_features_computes_box_score_rates(self) -> None:
        raw = pd.DataFrame(
            {
                "PLAYER_ID": [1],
                "PLAYER_NAME": ["Sample Guard"],
                "TEAM_ID": [100],
                "TEAM_ABBREVIATION": ["TST"],
                "AGE": [25],
                "GP": [10],
                "MIN": [360],
                "FGM": [80],
                "FGA": [160],
                "FG3M": [30],
                "FG3A": [80],
                "FTA": [40],
                "OREB": [10],
                "DREB": [30],
                "REB": [40],
                "AST": [60],
                "TOV": [20],
                "STL": [15],
                "BLK": [5],
                "PF": [25],
                "PTS": [230],
            }
        )

        features = build_player_season_features(raw, season="2023-24")
        row = features.iloc[0]

        self.assertEqual(row["season"], "2023-24")
        self.assertEqual(row["player_name"], "Sample Guard")
        self.assertEqual(row["minutes_per_game"], 36.0)
        self.assertEqual(row["points_per_36"], 23.0)
        self.assertEqual(row["assists_per_36"], 6.0)
        self.assertEqual(row["three_point_attempt_rate"], 0.5)
        self.assertEqual(row["free_throw_attempt_rate"], 0.25)
        self.assertEqual(row["effective_fg_pct"], 0.59375)
        self.assertAlmostEqual(row["true_shooting_pct"], 230 / (2 * (160 + 0.44 * 40)))
        self.assertEqual(row["assist_to_turnover"], 3.0)

    def test_build_player_season_features_handles_zero_denominators(self) -> None:
        raw = pd.DataFrame(
            {
                "PLAYER_ID": [1],
                "PLAYER_NAME": ["No Minutes"],
                "TEAM_ID": [100],
                "TEAM_ABBREVIATION": ["TST"],
                "AGE": [25],
                "GP": [0],
                "MIN": [0],
                "FGM": [0],
                "FGA": [0],
                "FG3M": [0],
                "FG3A": [0],
                "FTA": [0],
                "OREB": [0],
                "DREB": [0],
                "REB": [0],
                "AST": [0],
                "TOV": [0],
                "STL": [0],
                "BLK": [0],
                "PF": [0],
                "PTS": [0],
            }
        )

        features = build_player_season_features(raw, season="2023-24")

        self.assertTrue((features[FINGERPRINT_FEATURE_COLUMNS] == 0.0).all(axis=None))

    def test_build_player_season_features_filters_by_min_minutes(self) -> None:
        raw = pd.DataFrame(
            {
                "PLAYER_ID": [1, 2],
                "PLAYER_NAME": ["Low Minutes", "Rotation Player"],
                "TEAM_ID": [100, 100],
                "TEAM_ABBREVIATION": ["TST", "TST"],
                "AGE": [21, 26],
                "GP": [3, 82],
                "MIN": [20, 2000],
                "FGM": [1, 400],
                "FGA": [5, 900],
                "FG3M": [0, 100],
                "FG3A": [2, 300],
                "FTA": [0, 200],
                "OREB": [1, 80],
                "DREB": [1, 240],
                "REB": [2, 320],
                "AST": [0, 250],
                "TOV": [1, 120],
                "STL": [0, 70],
                "BLK": [0, 30],
                "PF": [1, 180],
                "PTS": [2, 1100],
            }
        )

        features = build_player_season_features(raw, season="2023-24", min_minutes=500)

        self.assertEqual(features["player_name"].tolist(), ["Rotation Player"])


if __name__ == "__main__":
    unittest.main()

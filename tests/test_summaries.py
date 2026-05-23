import unittest

import pandas as pd

from nba_fingerprints.models.summaries import build_player_summary_table


class SummaryTests(unittest.TestCase):
    def test_build_player_summary_table_combines_top_outputs(self) -> None:
        features = pd.DataFrame(
            {
                "season": ["2023-24"],
                "player_id": [1],
                "player_name": ["Sample Player"],
                "team_abbreviation": ["AAA"],
                "position": ["G"],
                "primary_position": ["G"],
                "games_played": [82],
                "minutes_total": [2500],
                "minutes_per_game": [30.5],
                "usage_rate": [0.25],
                "player_impact_estimate": [0.12],
            }
        )
        neighbors = pd.DataFrame(
            {
                "season": ["2023-24"],
                "player_id": [1],
                "team_abbreviation": ["AAA"],
                "neighbor_rank": [1],
                "neighbor_player_name": ["Similar Player"],
                "cosine_similarity": [0.95],
            }
        )
        position_scores = pd.DataFrame(
            {
                "season": ["2023-24"],
                "player_id": [1],
                "team_abbreviation": ["AAA"],
                "reference_position": ["G"],
                "position_rank": [1],
                "cosine_similarity": [0.9],
            }
        )
        archetype_scores = pd.DataFrame(
            {
                "season": ["2023-24"],
                "player_id": [1],
                "team_abbreviation": ["AAA"],
                "archetype": ["high_usage_creator"],
                "archetype_rank": [1],
                "cosine_similarity": [0.88],
            }
        )
        explanations = pd.DataFrame(
            {
                "season": ["2023-24"],
                "player_id": [1],
                "team_abbreviation": ["AAA"],
                "supporting_features": ["usage_rate=0.800"],
                "gap_features": ["assist_pct=0.200"],
            }
        )

        summary = build_player_summary_table(features, neighbors, position_scores, archetype_scores, explanations)
        row = summary.iloc[0]

        self.assertEqual(row["top_neighbor"], "Similar Player")
        self.assertEqual(row["top_position_reference"], "G")
        self.assertEqual(row["top_archetype"], "high_usage_creator")
        self.assertEqual(row["supporting_features"], "usage_rate=0.800")


if __name__ == "__main__":
    unittest.main()

import unittest

import pandas as pd

from nba_fingerprints.features.positions import (
    attach_position_labels,
    build_position_references,
    primary_position,
    score_position_similarity,
)


class PositionTests(unittest.TestCase):
    def test_primary_position_normalizes_combo_positions(self) -> None:
        self.assertEqual(primary_position("G"), "G")
        self.assertEqual(primary_position("G-F"), "G")
        self.assertEqual(primary_position("F-C"), "F")
        self.assertEqual(primary_position(" C "), "C")
        self.assertEqual(primary_position(None), "UNK")

    def test_attach_position_labels_joins_by_player_and_team(self) -> None:
        features = pd.DataFrame(
            {
                "player_id": [1, 2],
                "team_id": [100, 200],
                "player_name": ["Guard", "Big"],
            }
        )
        index = pd.DataFrame(
            {
                "PERSON_ID": [1, 2],
                "TEAM_ID": [0, 0],
                "POSITION": ["G", "C"],
            }
        )

        labeled = attach_position_labels(features, index)

        self.assertEqual(labeled["position"].tolist(), ["G", "C"])
        self.assertEqual(labeled["primary_position"].tolist(), ["G", "C"])

    def test_build_position_references_averages_fingerprints(self) -> None:
        fingerprints = pd.DataFrame(
            {
                "primary_position": ["G", "G", "C"],
                "creation": [1.0, 0.8, 0.0],
                "rim_profile": [0.0, 0.2, 1.0],
            }
        )

        references = build_position_references(fingerprints, feature_columns=["creation", "rim_profile"])
        pg = references[references["reference_position"] == "G"].iloc[0]

        self.assertEqual(pg["player_count"], 2)
        self.assertAlmostEqual(pg["creation"], 0.9)
        self.assertAlmostEqual(pg["rim_profile"], 0.1)

    def test_score_position_similarity_ranks_closest_reference(self) -> None:
        fingerprints = pd.DataFrame(
            {
                "season": ["2023-24"],
                "player_id": [1],
                "player_name": ["Creator"],
                "team_abbreviation": ["AAA"],
                "position": ["G"],
                "primary_position": ["G"],
                "creation": [1.0],
                "rim_profile": [0.0],
            }
        )
        references = pd.DataFrame(
            {
                "reference_position": ["G", "C"],
                "creation": [1.0, 0.0],
                "rim_profile": [0.0, 1.0],
            }
        )

        scores = score_position_similarity(fingerprints, references, feature_columns=["creation", "rim_profile"])

        self.assertEqual(scores.iloc[0]["reference_position"], "G")
        self.assertEqual(scores.iloc[0]["position_rank"], 1)


if __name__ == "__main__":
    unittest.main()

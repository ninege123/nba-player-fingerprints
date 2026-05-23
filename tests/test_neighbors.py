import unittest

import pandas as pd

from nba_fingerprints.models.neighbors import find_nearest_neighbors, player_similarity_matrix


class NeighborTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fingerprints = pd.DataFrame(
            {
                "season": ["2023-24", "2023-24", "2023-24"],
                "player_id": [1, 2, 3],
                "player_name": ["Creator A", "Creator B", "Big C"],
                "team_abbreviation": ["AAA", "BBB", "CCC"],
                "creation": [1.0, 0.9, 0.0],
                "rim_profile": [0.0, 0.1, 1.0],
            }
        )

    def test_player_similarity_matrix_returns_square_matrix(self) -> None:
        matrix = player_similarity_matrix(self.fingerprints, feature_columns=["creation", "rim_profile"])

        self.assertEqual(matrix.shape, (3, 3))
        self.assertAlmostEqual(matrix.loc["Creator A", "Creator A"], 1.0)
        self.assertAlmostEqual(matrix.loc["Creator A", "Big C"], 0.0)

    def test_find_nearest_neighbors_ranks_most_similar_player(self) -> None:
        neighbors = find_nearest_neighbors(
            self.fingerprints,
            top_n=1,
            feature_columns=["creation", "rim_profile"],
        )

        creator_a = neighbors[neighbors["player_name"] == "Creator A"].iloc[0]

        self.assertEqual(creator_a["neighbor_player_name"], "Creator B")
        self.assertEqual(creator_a["neighbor_rank"], 1)
        self.assertGreater(creator_a["cosine_similarity"], 0.99)

    def test_find_nearest_neighbors_rejects_invalid_top_n(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_n"):
            find_nearest_neighbors(self.fingerprints, top_n=0, feature_columns=["creation", "rim_profile"])


if __name__ == "__main__":
    unittest.main()

import unittest

import pandas as pd

from nba_fingerprints.features.archetypes import build_archetype_references, score_archetype_similarity


class ArchetypeTests(unittest.TestCase):
    def test_build_archetype_references_fills_missing_weights_with_zero(self) -> None:
        references = build_archetype_references(
            {"creator": {"points_per_36": 1.0}},
            feature_columns=["points_per_36", "assists_per_36"],
        )

        self.assertEqual(references.loc[0, "archetype"], "creator")
        self.assertEqual(references.loc[0, "points_per_36"], 1.0)
        self.assertEqual(references.loc[0, "assists_per_36"], 0.0)

    def test_build_archetype_references_rejects_unknown_features(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown feature weights"):
            build_archetype_references(
                {"bad": {"not_a_feature": 1.0}},
                feature_columns=["points_per_36"],
            )

    def test_score_archetype_similarity_ranks_matching_profile(self) -> None:
        fingerprints = pd.DataFrame(
            {
                "season": ["2023-24"],
                "player_id": [1],
                "player_name": ["Scorer"],
                "team_abbreviation": ["AAA"],
                "points_per_36": [1.0],
                "assists_per_36": [0.0],
            }
        )
        references = pd.DataFrame(
            {
                "archetype": ["scorer", "passer"],
                "points_per_36": [1.0, 0.0],
                "assists_per_36": [0.0, 1.0],
            }
        )

        scores = score_archetype_similarity(
            fingerprints,
            references,
            feature_columns=["points_per_36", "assists_per_36"],
        )

        self.assertEqual(scores.iloc[0]["archetype"], "scorer")
        self.assertEqual(scores.iloc[0]["archetype_rank"], 1)


if __name__ == "__main__":
    unittest.main()

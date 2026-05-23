import unittest

import pandas as pd

from nba_fingerprints.features.fingerprints import build_fingerprint_table


class FingerprintTests(unittest.TestCase):
    def test_build_fingerprint_table_keeps_metadata_and_scales_features(self) -> None:
        player_features = pd.DataFrame(
            {
                "season": ["2023-24", "2023-24"],
                "player_id": [1, 2],
                "player_name": ["Low Usage", "High Usage"],
                "team_abbreviation": ["AAA", "BBB"],
                "points_per_36": [10.0, 30.0],
                "assists_per_36": [5.0, 5.0],
            }
        )

        fingerprints = build_fingerprint_table(
            player_features,
            feature_columns=["points_per_36", "assists_per_36"],
        )

        self.assertEqual(fingerprints.columns.tolist(), ["season", "player_id", "player_name", "team_abbreviation", "points_per_36", "assists_per_36"])
        self.assertEqual(fingerprints["points_per_36"].tolist(), [0.0, 1.0])
        self.assertEqual(fingerprints["assists_per_36"].tolist(), [0.0, 0.0])

    def test_build_fingerprint_table_requires_metadata_and_features(self) -> None:
        player_features = pd.DataFrame({"player_name": ["Missing Columns"]})

        with self.assertRaisesRegex(ValueError, "player features is missing required columns"):
            build_fingerprint_table(player_features, feature_columns=["points_per_36"])


if __name__ == "__main__":
    unittest.main()

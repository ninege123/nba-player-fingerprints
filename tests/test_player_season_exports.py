import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from nba_fingerprints.pipelines.player_season_exports import (
    build_player_season_export_tables,
    export_player_season_tables,
    player_season_export_paths,
)


class PlayerSeasonExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw_stats = pd.DataFrame(
            {
                "PLAYER_ID": [1, 2, 3],
                "PLAYER_NAME": ["Creator A", "Creator B", "Big C"],
                "TEAM_ID": [100, 200, 300],
                "TEAM_ABBREVIATION": ["AAA", "BBB", "CCC"],
                "AGE": [25, 26, 27],
                "GP": [10, 10, 10],
                "MIN": [360, 360, 360],
                "FGM": [80, 78, 70],
                "FGA": [160, 158, 120],
                "FG3M": [30, 28, 2],
                "FG3A": [80, 76, 8],
                "FTA": [40, 38, 80],
                "OREB": [10, 11, 40],
                "DREB": [30, 28, 80],
                "REB": [40, 39, 120],
                "AST": [60, 58, 20],
                "TOV": [20, 19, 25],
                "STL": [15, 14, 8],
                "BLK": [5, 4, 30],
                "PF": [25, 24, 35],
                "PTS": [230, 222, 222],
            }
        )

    def test_player_season_export_paths_uses_stable_names(self) -> None:
        paths = player_season_export_paths("2023-24", output_dir="processed-test")

        self.assertEqual(paths.features, Path("processed-test/player_season_features_2023_24.csv"))
        self.assertEqual(paths.fingerprints, Path("processed-test/player_fingerprints_2023_24.csv"))
        self.assertEqual(paths.neighbors, Path("processed-test/player_neighbors_2023_24.csv"))

    def test_build_player_season_export_tables_returns_expected_tables(self) -> None:
        (
            features,
            fingerprints,
            neighbors,
            position_references,
            position_scores,
            archetype_references,
            archetype_scores,
            archetype_explanations,
        ) = build_player_season_export_tables(
            self.raw_stats,
            season="2023-24",
            player_index=pd.DataFrame(
                {
                    "PERSON_ID": [1, 2, 3],
                    "TEAM_ID": [100, 200, 300],
                    "POSITION": ["G", "G", "C"],
                }
            ),
            min_minutes=0,
            top_n=2,
        )

        self.assertEqual(features.shape[0], 3)
        self.assertEqual(fingerprints.shape[0], 3)
        self.assertEqual(neighbors.shape[0], 6)
        self.assertEqual(position_references.shape[0], 2)
        self.assertEqual(position_scores.shape[0], 6)
        self.assertEqual(archetype_references.shape[0], 8)
        self.assertEqual(archetype_scores.shape[0], 24)
        self.assertEqual(archetype_explanations.shape[0], 3)

    def test_export_player_season_tables_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            player_index = pd.DataFrame(
                {
                    "PERSON_ID": [1, 2, 3],
                    "TEAM_ID": [100, 200, 300],
                    "POSITION": ["G", "G", "C"],
                }
            )
            with patch("nba_fingerprints.pipelines.player_season_exports.load_player_season_stats", return_value=self.raw_stats):
                with patch("nba_fingerprints.pipelines.player_season_exports.load_player_index", return_value=player_index):
                    paths = export_player_season_tables(
                        "2023-24",
                        min_minutes=0,
                        top_n=1,
                        output_dir=tmp_dir,
                    )

            self.assertTrue(paths.features.exists())
            self.assertTrue(paths.fingerprints.exists())
            self.assertTrue(paths.neighbors.exists())
            self.assertTrue(paths.position_references.exists())
            self.assertTrue(paths.position_scores.exists())
            self.assertTrue(paths.archetype_references.exists())
            self.assertTrue(paths.archetype_scores.exists())
            self.assertTrue(paths.archetype_explanations.exists())
            self.assertEqual(pd.read_csv(paths.neighbors).shape[0], 3)


if __name__ == "__main__":
    unittest.main()

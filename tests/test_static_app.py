import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nba_fingerprints.app.static_app import generate_static_app


class StaticAppTests(unittest.TestCase):
    def test_generate_static_app_writes_html_with_embedded_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            processed_dir = Path(tmp_dir) / "processed"
            output_dir = Path(tmp_dir) / "site" / "app"
            processed_dir.mkdir()

            pd.DataFrame(
                {
                    "season": ["2023-24"],
                    "player_id": [1],
                    "player_name": ["Sample Player"],
                    "team_abbreviation": ["AAA"],
                    "primary_position": ["G"],
                    "top_archetype": ["creator"],
                }
            ).to_csv(processed_dir / "app_player_profiles_2023_24.csv", index=False)
            pd.DataFrame(
                {
                    "season": ["2023-24"],
                    "player_id": [1],
                    "player_name": ["Sample Player"],
                    "team_abbreviation": ["AAA"],
                    "neighbor_rank": [1],
                    "neighbor_player_id": [2],
                    "neighbor_player_name": ["Neighbor"],
                    "neighbor_team_abbreviation": ["BBB"],
                    "cosine_similarity": [0.9],
                }
            ).to_csv(processed_dir / "app_similarity_edges_2023_24.csv", index=False)
            pd.DataFrame(
                {
                    "feature": ["points_per_36"],
                    "label": ["Points Per 36"],
                    "group": ["scoring"],
                    "description": ["Scoring volume"],
                    "higher_is": ["more"],
                    "is_fingerprint_dimension": [True],
                }
            ).to_csv(processed_dir / "app_feature_metadata_2023_24.csv", index=False)

            path = generate_static_app("2023-24", processed_dir=processed_dir, output_dir=output_dir)

            self.assertTrue(path.exists())
            html = path.read_text(encoding="utf-8")
            self.assertIn("NBA Player Fingerprints App", html)
            self.assertIn("Sample Player", html)


if __name__ == "__main__":
    unittest.main()

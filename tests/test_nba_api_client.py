import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from nba_fingerprints.data.cache import cache_path, write_cached_frame
from nba_fingerprints.data.nba_api_client import load_player_season_stats


class NbaApiClientTests(unittest.TestCase):
    def test_load_player_season_stats_uses_existing_cache(self) -> None:
        cached = pd.DataFrame(
            {
                "PLAYER_ID": [1],
                "PLAYER_NAME": ["Sample Player"],
                "TEAM_ID": [100],
                "TEAM_ABBREVIATION": ["TST"],
                "AGE": [25],
                "GP": [82],
                "MIN": [2500],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = cache_path("player_season_stats_base_totals", "2023-24", cache_dir=tmp_dir, file_format="csv")
            write_cached_frame(cached, path)

            with patch("nba_fingerprints.data.nba_api_client.fetch_player_season_stats") as fetch:
                loaded = load_player_season_stats("2023-24", cache_dir=tmp_dir, file_format="csv")

        fetch.assert_not_called()
        pd.testing.assert_frame_equal(loaded, cached)

    def test_load_player_season_stats_writes_cache_after_fetch(self) -> None:
        fetched = pd.DataFrame(
            {
                "PLAYER_ID": [2],
                "PLAYER_NAME": ["Fetched Player"],
                "TEAM_ID": [200],
                "TEAM_ABBREVIATION": ["NEW"],
                "AGE": [26],
                "GP": [70],
                "MIN": [2100],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = cache_path("player_season_stats_base_totals", "2023-24", cache_dir=tmp_dir, file_format="csv")

            with patch("nba_fingerprints.data.nba_api_client.fetch_player_season_stats", return_value=fetched):
                loaded = load_player_season_stats("2023-24", cache_dir=tmp_dir, file_format="csv")

            self.assertTrue(Path(path).exists())

        pd.testing.assert_frame_equal(loaded, fetched)


if __name__ == "__main__":
    unittest.main()

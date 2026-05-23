import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nba_fingerprints.data.cache import cache_path, cached_frame_exists, read_cached_frame, write_cached_frame


class CacheTests(unittest.TestCase):
    def test_cache_path_uses_stable_season_filename(self) -> None:
        path = cache_path("player_season_stats_base_totals", "2023-24", cache_dir="tmp-cache", file_format="csv")

        self.assertEqual(path, Path("tmp-cache/player_season_stats_base_totals_2023_24.csv"))

    def test_cache_round_trip_csv(self) -> None:
        frame = pd.DataFrame({"PLAYER_ID": [1, 2], "PLAYER_NAME": ["A", "B"]})

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "players.csv"
            write_cached_frame(frame, path)
            self.assertTrue(cached_frame_exists(path))
            loaded = read_cached_frame(path)

        pd.testing.assert_frame_equal(loaded, frame)

    def test_cache_rejects_unknown_format(self) -> None:
        with self.assertRaisesRegex(ValueError, "file_format"):
            cache_path("players", "2023-24", file_format="json")


if __name__ == "__main__":
    unittest.main()

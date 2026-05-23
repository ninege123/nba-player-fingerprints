import unittest

import pandas as pd

from nba_fingerprints.features.schema import missing_columns, validate_columns


class SchemaTests(unittest.TestCase):
    def test_missing_columns_returns_absent_required_columns(self) -> None:
        frame = pd.DataFrame({"PLAYER_ID": [1], "PLAYER_NAME": ["A"]})

        self.assertEqual(missing_columns(frame, ["PLAYER_ID", "GP", "MIN"]), ["GP", "MIN"])

    def test_validate_columns_raises_clear_error(self) -> None:
        frame = pd.DataFrame({"PLAYER_ID": [1]})

        with self.assertRaisesRegex(ValueError, "test data is missing required columns"):
            validate_columns(frame, ["PLAYER_ID", "PLAYER_NAME"], dataset_name="test data")


if __name__ == "__main__":
    unittest.main()

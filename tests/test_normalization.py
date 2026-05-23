import pandas as pd
import unittest

from nba_fingerprints.features.normalization import min_max_scale_features, normalize_vector


class NormalizationTests(unittest.TestCase):
    def test_min_max_scale_features_scales_selected_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "player": ["A", "B", "C"],
                "usage_rate": [10.0, 20.0, 30.0],
                "assist_rate": [5.0, 5.0, 5.0],
            }
        )

        scaled = min_max_scale_features(frame, ["usage_rate", "assist_rate"])

        self.assertEqual(scaled["usage_rate"].tolist(), [0.0, 0.5, 1.0])
        self.assertEqual(scaled["assist_rate"].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(scaled["player"].tolist(), ["A", "B", "C"])

    def test_min_max_scale_features_rejects_missing_columns(self) -> None:
        frame = pd.DataFrame({"usage_rate": [10.0]})

        with self.assertRaisesRegex(KeyError, "Missing feature columns"):
            min_max_scale_features(frame, ["usage_rate", "assist_rate"])

    def test_normalize_vector_handles_nonzero_and_zero_vectors(self) -> None:
        self.assertEqual(normalize_vector([3.0, 4.0]).round(3).tolist(), [0.6, 0.8])
        self.assertEqual(normalize_vector([0.0, 0.0]).tolist(), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()

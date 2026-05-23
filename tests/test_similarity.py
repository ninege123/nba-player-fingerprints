import unittest

from nba_fingerprints.models.similarity import cosine_similarity


class SimilarityTests(unittest.TestCase):
    def test_cosine_similarity_scores_matching_and_orthogonal_vectors(self) -> None:
        self.assertEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_cosine_similarity_returns_zero_for_empty_profile(self) -> None:
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 1.0]), 0.0)


if __name__ == "__main__":
    unittest.main()

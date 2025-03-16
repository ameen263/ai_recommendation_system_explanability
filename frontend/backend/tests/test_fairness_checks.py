import unittest
from backend.fairness_checks import (
    popularity_bias_score,
    diversity_score,
    exposure_fairness_score,
    check_bias_and_fairness,
    load_ratings,
    load_movies
)
import pandas as pd

class FairnessChecksTests(unittest.TestCase):

    def setUp(self):
        """Setup method to load movies and ratings data, and define dummy recommendations."""
        self.movies = load_movies("data/u.item")
        self.ratings = load_ratings("data/u.data")
        # Define dummy recommendations for testing as a list of movie IDs
        self.recommendations = [1, 2, 3]

    def test_popularity_bias_score(self):
        """Test popularity bias calculation."""
        score = popularity_bias_score(self.recommendations)
        self.assertIsInstance(score, float)
        # Assuming the score is normalized, it should be between 0 and 1.
        self.assertTrue(0 <= score <= 1)

    def test_diversity_score(self):
        """Test diversity score calculation."""
        score = diversity_score(self.recommendations)
        self.assertIsInstance(score, float)
        # Diversity should be a positive value.
        self.assertTrue(score > 0)

    def test_exposure_fairness_score(self):
        """Test exposure fairness calculation."""
        score = exposure_fairness_score(self.recommendations)
        self.assertIsInstance(score, float)
        # Exposure fairness (coefficient of variation) should be non-negative.
        self.assertTrue(score >= 0)

    def test_check_bias_and_fairness(self):
        """Test overall bias and fairness check returns the expected keys."""
        metrics = check_bias_and_fairness(self.recommendations)
        self.assertIsInstance(metrics, dict)
        self.assertIn("Popularity Bias Score", metrics)
        self.assertIn("Diversity Score", metrics)
        self.assertIn("Exposure Fairness Score", metrics)

    def test_load_ratings_success(self):
        """Test loading ratings data successfully."""
        self.assertIsInstance(self.ratings, pd.DataFrame)
        self.assertFalse(self.ratings.empty)
        # Verify that the expected columns are present in the ratings data.
        self.assertIn("userId", self.ratings.columns)
        self.assertIn("movieId", self.ratings.columns)
        self.assertIn("rating", self.ratings.columns)
        self.assertIn("timestamp", self.ratings.columns)

    def test_load_movies_success(self):
        """Test loading movies data successfully."""
        self.assertIsInstance(self.movies, pd.DataFrame)
        self.assertFalse(self.movies.empty)
        # Verify that the expected columns are present in the movies data.
        self.assertIn("movie_id", self.movies.columns)
        self.assertIn("title", self.movies.columns)
        self.assertIn("genres", self.movies.columns)

if __name__ == '__main__':
    unittest.main()

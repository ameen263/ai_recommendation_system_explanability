import unittest
from unittest.mock import patch
from backend.hybrid_recommend import HybridRecommender, ContentBasedRecommender
import pandas as pd
from surprise import SVD

class HybridRecommenderTests(unittest.TestCase):

    def setUp(self):
        """Setup method to create a HybridRecommender instance."""
        self.recommender = HybridRecommender()

    @patch('backend.hybrid_recommend.pd.read_csv')
    def test_load_ratings_success(self, mock_read_csv):
        """Test loading ratings data successfully."""
        mock_read_csv.return_value = pd.DataFrame({
            'user_id': [1, 2, 3],
            'movie_id': [101, 102, 103],
            'rating': [5, 4, 3],
            'timestamp': [1111111111, 1111111112, 1111111113]
        })
        ratings = self.recommender.load_ratings()
        self.assertIsInstance(ratings, pd.DataFrame)
        self.assertFalse(ratings.empty)

    @patch('backend.hybrid_recommend.pd.read_csv')
    def test_load_ratings_file_not_found(self, mock_read_csv):
        """Test loading ratings data when the file does not exist."""
        mock_read_csv.side_effect = FileNotFoundError
        ratings = self.recommender.load_ratings()
        self.assertIsNone(ratings)

    @patch('backend.hybrid_recommend.pd.read_csv')
    def test_load_movies_success(self, mock_read_csv):
        """Test loading movies data successfully."""
        mock_read_csv.return_value = pd.DataFrame({
            'movie_id': [101, 102, 103],
            'title': ['Movie 1', 'Movie 2', 'Movie 3']
        })
        movies = self.recommender.load_movies()
        self.assertIsInstance(movies, pd.DataFrame)
        self.assertFalse(movies.empty)

    @patch('backend.hybrid_recommend.pd.read_csv')
    def test_load_movies_file_not_found(self, mock_read_csv):
        """Test loading movies data when the file does not exist."""
        mock_read_csv.side_effect = FileNotFoundError
        movies = self.recommender.load_movies()
        self.assertIsNone(movies)

    @patch('builtins.open')
    @patch('backend.hybrid_recommend.pickle.load')
    def test_load_model_success(self, mock_pickle_load, mock_open):
        """Test loading the model successfully."""
        mock_pickle_load.return_value = SVD()
        model = self.recommender.load_model()
        self.assertIsInstance(model, SVD)

    @patch('builtins.open')
    @patch('backend.hybrid_recommend.pickle.load')
    def test_load_model_file_not_found(self, mock_pickle_load, mock_open):
        """Test loading the model when the file does not exist."""
        mock_open.side_effect = FileNotFoundError
        model = self.recommender.load_model()
        self.assertIsNone(model)

    @patch('backend.hybrid_recommend.HybridRecommender.get_cf_recommendations')
    def test_hybrid_recommendation(self, mock_get_cf_recommendations):
        """Test generating hybrid recommendations."""
        # Simulate CF recommendations as a list of tuples: (movie_id, cf_score)
        mock_get_cf_recommendations.return_value = [
            (1, 4.5), (2, 4.2), (3, 4.0)
        ]
        # Set dummy movies data
        self.recommender.movies = pd.DataFrame({
            'movie_id': [1, 2, 3],
            'title': ['Movie 1', 'Movie 2', 'Movie 3']
        })
        # Instantiate a dummy content-based recommender
        self.recommender.content_recommender = ContentBasedRecommender('data/u.item')

        recs = self.recommender.hybrid_recommendation(1)
        # Check that recommendations are returned as a list containing three items.
        self.assertIsInstance(recs, list)
        self.assertEqual(len(recs), 3)
        for rec in recs:
            # Each recommendation should be a tuple of (movie_id, title, score)
            self.assertIsInstance(rec, tuple)
            self.assertEqual(len(rec), 3)

if __name__ == '__main__':
    unittest.main()

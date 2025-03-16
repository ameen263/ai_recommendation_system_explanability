import unittest
import pandas as pd
from backend.content_filtering import ContentBasedRecommender  # Import from backend

class ContentFilteringTests(unittest.TestCase):

    def setUp(self):
        """Setup method to create a ContentBasedRecommender instance."""
        self.recommender = ContentBasedRecommender("data/u.item")

    def test_load_movies_success(self):
        """Test that movies data is loaded successfully and is not empty."""
        self.assertIsInstance(self.recommender.movies, pd.DataFrame)
        self.assertFalse(self.recommender.movies.empty)

    def test_load_movies_file_not_found(self):
        """Test that a non-existent movies file results in an empty DataFrame."""
        recommender = ContentBasedRecommender("non_existent_file.csv")
        self.assertTrue(recommender.movies.empty)

    def test_compute_similarity_matrix(self):
        """Test that the cosine similarity matrix is computed correctly."""
        self.assertIsNotNone(self.recommender.cosine_sim)
        # Check that the matrix shape matches the number of movies
        expected_shape = (len(self.recommender.movies), len(self.recommender.movies))
        self.assertEqual(self.recommender.cosine_sim.shape, expected_shape)

    def test_get_similar_movies_valid_movie_id(self):
        """Test retrieving similar movies for a valid movie ID returns a list of dictionaries."""
        similar_movies = self.recommender.get_similar_movies(movie_id=1, top_n=5)
        self.assertIsInstance(similar_movies, list)
        self.assertEqual(len(similar_movies), 5)
        # Verify each recommendation is a dictionary with expected keys
        for movie in similar_movies:
            self.assertIsInstance(movie, dict)
            self.assertIn('movie_id', movie)
            self.assertIn('title', movie)
            self.assertIn('similarity_score', movie)

    def test_get_similar_movies_invalid_movie_id(self):
        """Test that requesting similar movies for an invalid movie ID returns an empty list."""
        similar_movies = self.recommender.get_similar_movies(movie_id=9999999, top_n=5)
        self.assertIsInstance(similar_movies, list)
        self.assertEqual(len(similar_movies), 0)

    def test_get_movie_by_id_valid_id(self):
        """Test that retrieving a movie by a valid ID returns a dictionary with movie details."""
        movie = self.recommender.get_movie_by_id(1)
        self.assertIsInstance(movie, dict)
        self.assertIn('movie_id', movie)
        self.assertIn('title', movie)

    def test_get_movie_by_id_invalid_id(self):
        """Test that retrieving a movie by an invalid ID returns None."""
        movie = self.recommender.get_movie_by_id(9999999)
        self.assertIsNone(movie)

if __name__ == '__main__':
    unittest.main()

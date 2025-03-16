import unittest
from unittest.mock import patch
from backend.integrate_data import load_users, load_ratings, merge_datasets
import pandas as pd
import sqlite3

class IntegrateDataTests(unittest.TestCase):

    @patch('backend.integrate_data.sqlite3.connect')
    @patch('backend.integrate_data.pd.read_sql_query')
    def test_load_users_success(self, mock_read_sql_query, mock_connect):
        """Test loading users data successfully."""
        # Provide dummy user data
        mock_read_sql_query.return_value = pd.DataFrame({
            'id': [1, 2, 3],
            'username': ['user1', 'user2', 'user3']
        })
        users = load_users('test.db')
        self.assertIsInstance(users, pd.DataFrame)
        self.assertFalse(users.empty)
        # Optionally, check for expected columns
        self.assertIn('id', users.columns)
        self.assertIn('username', users.columns)

    @patch('backend.integrate_data.sqlite3.connect')
    @patch('backend.integrate_data.pd.read_sql_query')
    def test_load_users_db_error(self, mock_read_sql_query, mock_connect):
        """Test loading users data when a database error occurs."""
        mock_connect.side_effect = sqlite3.Error("Database error")
        users = load_users('test.db')
        self.assertIsNone(users)

    @patch('backend.integrate_data.pd.read_csv')
    def test_load_ratings_success(self, mock_read_csv):
        """Test loading ratings data successfully."""
        # Provide dummy ratings data
        mock_read_csv.return_value = pd.DataFrame({
            'user_id_old': [1, 2, 3],
            'movie_id': [101, 102, 103],
            'rating': [4.5, 3.0, 5.0],
            'timestamp': [1111111111, 1111111112, 1111111113]
        })
        ratings = load_ratings('test.csv')
        self.assertIsInstance(ratings, pd.DataFrame)
        self.assertFalse(ratings.empty)
        # Check that the expected columns are present
        self.assertIn('user_id_old', ratings.columns)
        self.assertIn('movie_id', ratings.columns)
        self.assertIn('rating', ratings.columns)
        self.assertIn('timestamp', ratings.columns)

    @patch('backend.integrate_data.pd.read_csv')
    def test_load_ratings_file_not_found(self, mock_read_csv):
        """Test loading ratings data when the file does not exist."""
        mock_read_csv.side_effect = FileNotFoundError
        ratings = load_ratings('test.csv')
        self.assertIsNone(ratings)

    def test_merge_datasets(self):
        """Test merging users and ratings data and check resulting DataFrame structure."""
        # Create dummy users DataFrame
        users = pd.DataFrame({
            'id': [1, 2, 3],
            'username': ['user1', 'user2', 'user3']
        })
        # Create dummy ratings DataFrame with multiple ratings for some users
        ratings = pd.DataFrame({
            'user_id_old': [1, 1, 2, 2],
            'movie_id': [101, 102, 103, 104],
            'rating': [4.5, 3.0, 5.0, 4.2],
            'timestamp': [1111111111, 1111111112, 1111111113, 1111111114]
        })
        merged_df, user_map = merge_datasets(users, ratings)
        self.assertIsInstance(merged_df, pd.DataFrame)
        self.assertFalse(merged_df.empty)
        self.assertIsInstance(user_map, dict)
        # Verify that the merged DataFrame has the renamed 'user_id' column instead of 'user_id_old'
        self.assertIn('user_id', merged_df.columns)
        self.assertNotIn('user_id_old', merged_df.columns)
        # Optionally, check that the number of rows is as expected (orphaned records dropped)
        self.assertGreaterEqual(len(merged_df), 1)

if __name__ == '__main__':
    unittest.main()

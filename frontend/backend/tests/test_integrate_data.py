import os
import sys
import unittest
from unittest.mock import patch
import pandas as pd

# Add the project root (two levels up) to sys.path so that the backend package is found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.integrate_data import load_users, load_ratings, merge_datasets

def dummy_read_csv(*args, **kwargs):
    """
    A helper function that simulates pd.read_csv behavior by returning a DataFrame with the
    columns specified in the 'names' keyword argument. The data is hard-coded.
    """
    data = [
        [1, 101, 4.5, 1111111111],
        [2, 102, 3.0, 1111111112],
        [3, 103, 5.0, 1111111113]
    ]
    col_names = kwargs.get('names')
    return pd.DataFrame(data, columns=col_names)

class IntegrateDataTests(unittest.TestCase):

    @patch('backend.integrate_data.pd.read_csv')
    def test_load_users_success(self, mock_read_csv):
        """Test loading users data successfully from a CSV file."""
        # Provide dummy user data with the expected 5-column format.
        dummy_users = pd.DataFrame({
            0: [1, 2, 3],
            1: [25, 30, 22],
            2: ['M', 'F', 'M'],
            3: ['engineer', 'artist', 'student'],
            4: ['12345', '67890', '54321']
        })
        mock_read_csv.return_value = dummy_users

        users = load_users('dummy_u.user')
        self.assertIsInstance(users, pd.DataFrame)
        self.assertFalse(users.empty)
        # Check that the columns are renamed correctly.
        self.assertIn('user_id', users.columns)
        self.assertIn('age', users.columns)
        self.assertIn('gender', users.columns)
        self.assertIn('occupation', users.columns)
        self.assertIn('zip_code', users.columns)

    @patch('backend.integrate_data.pd.read_csv')
    def test_load_users_file_error(self, mock_read_csv):
        """Test loading users data when a file error occurs."""
        # Simulate a FileNotFoundError.
        mock_read_csv.side_effect = FileNotFoundError("File not found")
        users = load_users('non_existent_u.user')
        self.assertIsNone(users)

    @patch('backend.integrate_data.pd.read_csv', side_effect=dummy_read_csv)
    def test_load_ratings_success(self, mock_read_csv):
        """Test loading ratings data successfully."""
        ratings = load_ratings('dummy_u.data')
        self.assertIsInstance(ratings, pd.DataFrame)
        self.assertFalse(ratings.empty)
        # Check that the DataFrame has the expected column names.
        self.assertListEqual(list(ratings.columns), ['user_id', 'movie_id', 'rating', 'timestamp'])

    @patch('backend.integrate_data.pd.read_csv')
    def test_load_ratings_file_not_found(self, mock_read_csv):
        """Test loading ratings data when the file does not exist."""
        mock_read_csv.side_effect = FileNotFoundError("File not found")
        ratings = load_ratings('non_existent_u.data')
        self.assertIsNone(ratings)

    def test_merge_datasets(self):
        """Test merging users and ratings data and check resulting DataFrame structure."""
        # Create dummy users DataFrame with expected column 'user_id'.
        users = pd.DataFrame({
            'user_id': [1, 2, 3],
            'username': ['user1', 'user2', 'user3']
        })
        # Create dummy ratings DataFrame using 'user_id_old' to simulate alternate naming.
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
        # Verify that the merged DataFrame has the standardized 'user_id' column.
        self.assertIn('user_id', merged_df.columns)
        # 'user_id_old' should not exist after the merge.
        self.assertNotIn('user_id_old', merged_df.columns)
        # Optionally, ensure that orphaned records were dropped.
        self.assertGreaterEqual(len(merged_df), 1)

if __name__ == '__main__':
    unittest.main()

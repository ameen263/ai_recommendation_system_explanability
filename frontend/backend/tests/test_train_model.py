import os
import sys
import unittest
from unittest.mock import patch
import pandas as pd

# Add the project root (two levels up) to sys.path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.train_model import load_data

class TrainModelTests(unittest.TestCase):

    @patch('backend.train_model.pd.read_csv')
    @patch('backend.train_model.Path.exists', return_value=True)
    def test_load_data_success(self, mock_exists, mock_read_csv):
        """Test loading ratings data successfully."""
        # Provide dummy data with camelCase column names to test renaming.
        dummy_data = pd.DataFrame({
            'userId': [1, 2, 3],
            'movieId': [10, 20, 30],
            'rating': [5.0, 3.5, 4.0],
            'timestamp': [1111111111, 1111111112, 1111111113]
        })
        mock_read_csv.return_value = dummy_data

        ratings = load_data('test.data')
        self.assertIsInstance(ratings, pd.DataFrame)
        self.assertFalse(ratings.empty)
        # Verify columns have been renamed to snake_case.
        self.assertIn('user_id', ratings.columns)
        self.assertIn('movie_id', ratings.columns)

    @patch('backend.train_model.pd.read_csv')
    @patch('backend.train_model.Path.exists', return_value=False)
    def test_load_data_file_not_found(self, mock_exists, mock_read_csv):
        """Test loading ratings data when the file does not exist."""
        # When the file doesn't exist, load_data should call sys.exit(1).
        with self.assertRaises(SystemExit) as cm:
            load_data('non_existent_file.data')
        self.assertEqual(cm.exception.code, 1)

if __name__ == '__main__':
    unittest.main()

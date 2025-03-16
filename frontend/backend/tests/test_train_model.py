import unittest
from unittest.mock import patch
from backend.train_model import load_data  # Import from backend
import pandas as pd

class TrainModelTests(unittest.TestCase):

    @patch('backend.train_model.pd.read_csv')  # Patch from backend
    def test_load_data_success(self, mock_read_csv):
        """Test loading ratings data successfully."""
        # Provide dummy data for the DataFrame columns.
        mock_read_csv.return_value = pd.DataFrame({
            'userId': [1, 2, 3],
            'movieId': [10, 20, 30],
            'rating': [5.0, 3.5, 4.0],
            'timestamp': [1111111111, 1111111112, 1111111113]
        })
        ratings = load_data('test.data')
        self.assertIsInstance(ratings, pd.DataFrame)
        self.assertFalse(ratings.empty)

    @patch('backend.train_model.pd.read_csv')  # Patch from backend
    def test_load_data_file_not_found(self, mock_read_csv):
        """Test loading ratings data when the file does not exist."""
        # Simulate a file not found error.
        mock_read_csv.side_effect = FileNotFoundError
        with self.assertRaises(SystemExit) as cm:  # Check for SystemExit
            load_data('non_existent_file.data')
        self.assertEqual(cm.exception.code, 1)  # Check exit code

if __name__ == '__main__':
    unittest.main()

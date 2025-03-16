import unittest
from unittest.mock import patch
from backend.evaluation import RecommenderEvaluator  # Import from backend
import pandas as pd
from surprise import SVD

class EvaluationTests(unittest.TestCase):
    def setUp(self):
        """Setup method to create a RecommenderEvaluator instance."""
        self.evaluator = RecommenderEvaluator()

    @patch('backend.evaluation.pd.read_csv')  # Patch from backend
    def test_load_data_success(self, mock_read_csv):
        """Test that loading ratings data successfully returns True and sets ratings_df."""
        # Provide valid dummy data for the DataFrame
        mock_read_csv.return_value = pd.DataFrame({
            'user_id': [1, 2, 3],
            'movie_id': [10, 20, 30],
            'rating': [4.0, 3.5, 5.0],
            'timestamp': [1111111111, 1111111112, 1111111113]
        })
        result = self.evaluator.load_data()
        self.assertTrue(result)
        self.assertIsInstance(self.evaluator.ratings_df, pd.DataFrame)
        self.assertFalse(self.evaluator.ratings_df.empty)

    @patch('backend.evaluation.pd.read_csv')  # Patch from backend
    def test_load_data_file_not_found(self, mock_read_csv):
        """Test that loading ratings data when the file does not exist raises FileNotFoundError."""
        mock_read_csv.side_effect = FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            self.evaluator.load_data()

    @patch('builtins.open')
    @patch('backend.evaluation.pickle.load')  # Patch from backend
    def test_load_model_success(self, mock_pickle_load, mock_open):
        """Test that loading the model successfully returns True and sets model."""
        mock_pickle_load.return_value = SVD()
        result = self.evaluator.load_model()
        self.assertTrue(result)
        self.assertIsInstance(self.evaluator.model, SVD)

    @patch('builtins.open')
    @patch('backend.evaluation.pickle.load')  # Patch from backend
    def test_load_model_file_not_found(self, mock_pickle_load, mock_open):
        """Test that loading the model when the file does not exist raises FileNotFoundError."""
        mock_open.side_effect = FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            self.evaluator.load_model()

    @patch('backend.evaluation.RecommenderEvaluator.compute_rmse')  # Patch from backend
    @patch('backend.evaluation.RecommenderEvaluator.compute_precision_recall')  # Patch from backend
    @patch('backend.evaluation.RecommenderEvaluator.compute_ndcg')  # Patch from backend
    @patch('backend.evaluation.RecommenderEvaluator.load_data')  # Patch from backend
    @patch('backend.evaluation.RecommenderEvaluator.load_model')  # Patch from backend
    def test_evaluate_model(self, mock_load_model, mock_load_data, mock_ndcg, mock_precision_recall, mock_rmse):
        """Test the evaluate_model method returns a dictionary with expected keys."""
        # Set the patched methods to return success and dummy metric values
        mock_load_model.return_value = True
        mock_load_data.return_value = True
        mock_rmse.return_value = 0.85
        mock_precision_recall.return_value = (0.75, 0.6)
        mock_ndcg.return_value = 0.8

        metrics = self.evaluator.evaluate_model()
        self.assertIsInstance(metrics, dict)
        self.assertIn('RMSE', metrics)
        self.assertIn('Precision@10', metrics)
        self.assertIn('Recall@10', metrics)
        self.assertIn('NDCG@10', metrics)

    def test_compute_rmse(self):
        """Test RMSE calculation returns a float value."""
        # Prepare test data and predictions (format: (user, item, actual, predicted, _))
        testset = [
            (1, 1, 4, 4.5, None),
            (1, 2, 3, 3.5, None),
            (2, 1, 5, 4, None)
        ]
        self.evaluator.model = SVD()  # Set a dummy model
        rmse = self.evaluator.compute_rmse(testset)
        self.assertIsInstance(rmse, float)

    def test_compute_precision_recall(self):
        """Test that precision and recall calculation returns float values."""
        # Prepare test data and predictions (format: (user, item, actual, predicted, _))
        predictions = [
            (1, 1, 4, 4.5, None),
            (1, 2, 3, 3.5, None),
            (2, 1, 5, 4, None)
        ]
        precision, recall = self.evaluator.compute_precision_recall(predictions, k=2)
        self.assertIsInstance(precision, float)
        self.assertIsInstance(recall, float)

    def test_compute_ndcg(self):
        """Test that NDCG calculation returns a float value."""
        # Prepare test data and predictions (format: (user, item, actual, predicted, _))
        predictions = [
            (1, 1, 4, 4.5, None),
            (1, 2, 3, 3.5, None),
            (2, 1, 5, 4, None)
        ]
        ndcg = self.evaluator.compute_ndcg(predictions, k=2)
        self.assertIsInstance(ndcg, float)

if __name__ == '__main__':
    unittest.main()

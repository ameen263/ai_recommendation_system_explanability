import unittest
from unittest.mock import patch, MagicMock
from backend.explainability import RecommendationExplainer, ContentBasedRecommender
import pandas as pd
from surprise import SVD

class ExplainabilityTests(unittest.TestCase):

    def setUp(self):
        """Setup method to create a RecommendationExplainer instance."""
        self.explainer = RecommendationExplainer()

    @patch('builtins.open')
    @patch('backend.explainability.pickle.load')
    def test_load_model_success(self, mock_pickle_load, mock_open):
        """Test loading the model successfully."""
        # Simulate successful loading by returning a dummy SVD model.
        mock_pickle_load.return_value = SVD()
        self.explainer._load_model()  # Call the method directly
        self.assertIsInstance(self.explainer.svd_model, SVD)

    @patch('builtins.open')
    @patch('backend.explainability.pickle.load')
    def test_load_model_file_not_found(self, mock_pickle_load, mock_open):
        """Test loading the model when the file does not exist."""
        # Simulate file not found by having open() raise FileNotFoundError.
        mock_open.side_effect = FileNotFoundError
        self.explainer._load_model()  # Call the method directly
        self.assertIsNone(self.explainer.svd_model)

    @patch('backend.explainability.pd.read_csv')
    def test_load_ratings_success(self, mock_read_csv):
        """Test loading ratings data successfully."""
        # Provide valid dummy data for the DataFrame.
        mock_read_csv.return_value = pd.DataFrame({
            'user_id': [1, 2, 3],
            'movie_id': [10, 20, 30],
            'rating': [4.0, 3.5, 5.0],
            'timestamp': [1111111111, 1111111112, 1111111113]
        })
        self.explainer._load_ratings()  # Call the method directly
        self.assertIsInstance(self.explainer.ratings_data, pd.DataFrame)
        self.assertFalse(self.explainer.ratings_data.empty)

    @patch('backend.explainability.pd.read_csv')
    def test_load_ratings_file_not_found(self, mock_read_csv):
        """Test loading ratings data when the file does not exist."""
        mock_read_csv.side_effect = FileNotFoundError
        self.explainer._load_ratings()  # Call the method directly
        self.assertIsNone(self.explainer.ratings_data)

    @patch('backend.explainability.lime.lime_tabular.LimeTabularExplainer')
    def test_explain_with_lime(self, mock_lime_explainer):
        """Test LIME explanation."""
        # Create a dummy explanation object with an as_list() method.
        mock_explanation = MagicMock()
        mock_explanation.as_list.return_value = [('feature1', 0.5), ('feature2', -0.2)]
        # Configure the LimeTabularExplainer instance to return the dummy explanation.
        mock_instance = mock_lime_explainer.return_value
        mock_instance.explain_instance.return_value = mock_explanation

        # Set a dummy SVD model.
        self.explainer.svd_model = SVD()
        # Provide dummy ratings data.
        self.explainer.ratings_data = pd.DataFrame({
            'user_id': [1, 2, 3],
            'movie_id': [10, 20, 30],
            'rating': [4.0, 3.5, 5.0],
            'timestamp': [1111111111, 1111111112, 1111111113]
        })
        exp = self.explainer.explain_with_lime(1, 1)
        self.assertEqual(exp, [('feature1', 0.5), ('feature2', -0.2)])

    @patch('backend.explainability.shap.KernelExplainer')
    def test_explain_with_shap(self, mock_shap_explainer):
        """Test SHAP explanation."""
        # Simulate dummy SHAP values.
        dummy_shap_values = [[0.3, 0.7]]  # Example: two features' contributions
        mock_instance = mock_shap_explainer.return_value
        mock_instance.shap_values.return_value = dummy_shap_values

        self.explainer.svd_model = SVD()  # Set a dummy model
        exp = self.explainer.explain_with_shap(1, 1)
        expected_output = {'user_contribution': 0.3, 'movie_contribution': 0.7}
        self.assertEqual(exp, expected_output)

    @patch('backend.explainability.ContentBasedRecommender.get_similar_movies')
    def test_get_user_history_explanation(self, mock_get_similar_movies):
        """Test user history explanation."""
        # Simulate the content-based recommender returning similar movies.
        mock_get_similar_movies.return_value = [
            {'movie_id': 2, 'title': 'Movie 2', 'similarity_score': 0.8},
            {'movie_id': 3, 'title': 'Movie 3', 'similarity_score': 0.6}
        ]
        # Provide dummy ratings data.
        self.explainer.ratings_data = pd.DataFrame({
            'user_id': [1, 1, 2],
            'movie_id': [1, 2, 3],
            'rating': [5.0, 4.0, 3.0],
            'timestamp': [1111111111, 1111111112, 1111111113]
        })
        # Instantiate the content-based recommender.
        self.explainer.content_recommender = ContentBasedRecommender('data/u.item')
        exp = self.explainer.get_user_history_explanation(1, 1)
        self.assertIn('target_movie', exp)
        self.assertIn('similar_movies', exp)
        self.assertIn('user_highly_rated', exp)

if __name__ == '__main__':
    unittest.main()

import unittest

import torch
from frontend.backend.advanced_explainability import AdvancedExplainer


class AdvancedExplainerTests(unittest.TestCase):
    def test_initialization_with_model_and_baseline(self):
        """Test that AdvancedExplainer initializes correctly with a valid model and baseline."""
        model = torch.nn.Linear(10, 1)
        baseline = torch.zeros(1, 10)
        explainer = AdvancedExplainer(model=model, baseline=baseline)

        self.assertIsNotNone(explainer.model)
        self.assertIsNotNone(explainer.baseline)
        self.assertIsNotNone(explainer.integrated_gradients)

    def test_initialization_without_model(self):
        """Test that AdvancedExplainer initializes correctly without a model."""
        explainer = AdvancedExplainer(model=None, baseline=None)

        self.assertIsNone(explainer.model)
        self.assertIsNone(explainer.baseline)
        self.assertIsNone(explainer.integrated_gradients)

    def test_initialization_with_only_model(self):
        """Test that AdvancedExplainer initializes correctly with only a model."""
        model = torch.nn.Linear(10, 1)
        explainer = AdvancedExplainer(model=model, baseline=None)

        self.assertIsNotNone(explainer.model)
        self.assertIsNone(explainer.baseline)
        self.assertIsNotNone(explainer.integrated_gradients)

    def test_initialization_with_wrong_baseline_type(self):
        """Test that AdvancedExplainer raises an error when an invalid baseline type is provided."""
        model = torch.nn.Linear(10, 1)
        invalid_baseline = [0] * 10  # baseline must be a tensor
        with self.assertRaises(TypeError):
            AdvancedExplainer(model=model, baseline=invalid_baseline)


if __name__ == "__main__":
    unittest.main()

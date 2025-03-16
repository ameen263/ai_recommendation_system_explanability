import torch
import torch.nn as nn
import logging
import numpy as np
from captum.attr import IntegratedGradients

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class AdvancedExplainer:
    def __init__(self, model=None, baseline=None):
        """
        Initialize the advanced explainer using Integrated Gradients.

        Args:
            model (torch.nn.Module): A PyTorch model used for making predictions.
            baseline (torch.Tensor, optional): Baseline input for Integrated Gradients.
                                                Defaults to a zero tensor matching input shape.
        """
        if model is None:
            logger.warning("No model provided. Advanced explanations will be limited.")
        self.model = model
        self.baseline = baseline
        self.integrated_gradients = IntegratedGradients(self.model) if self.model is not None else None

    def get_explanation(self, user_id: int, movie_id: int) -> dict:
        """
        Generate an advanced explanation for the recommendation using Integrated Gradients.

        Args:
            user_id (int): The user ID for which the recommendation was made.
            movie_id (int): The movie ID for which explanation is needed.

        Returns:
            dict: A dictionary with attribution details and an interpretation.
        """
        if self.model is None or self.integrated_gradients is None:
            logger.error("Model not set for advanced explanations.")
            return {"error": "Model not provided for advanced explanations."}

        try:
            # Create a dummy input tensor for demonstration.
            # In practice, this should be your actual model input vector.
            input_tensor = torch.tensor([user_id, movie_id], dtype=torch.float).unsqueeze(0)
            # Use baseline of zeros if not provided
            baseline = self.baseline if self.baseline is not None else torch.zeros_like(input_tensor)

            # Compute attributions using Integrated Gradients.
            attributions, delta = self.integrated_gradients.attribute(
                input_tensor, baseline, target=0, return_convergence_delta=True
            )
            attributions = attributions.squeeze(0).detach().numpy().tolist()
            explanation = {
                "user_contribution": attributions[0],
                "movie_contribution": attributions[1],
                "convergence_delta": delta.item(),
                "interpretation": (
                    "Higher attribution values indicate a stronger influence of that input feature "
                    "on the predicted recommendation score."
                )
            }
            logger.info(f"Advanced explanation generated for user {user_id}, movie {movie_id}.")
            return explanation
        except Exception as e:
            logger.exception(f"Error generating advanced explanation: {e}")
            return {"error": str(e)}


if __name__ == "__main__":
    # Example dummy model: predicts the sum of inputs.
    class DummyModel(nn.Module):
        def forward(self, x):
            return x.sum(dim=1, keepdim=True)


    dummy_model = DummyModel()
    adv_explainer = AdvancedExplainer(model=dummy_model)
    explanation = adv_explainer.get_explanation(user_id=1, movie_id=1)
    print("Advanced Explanation:", explanation)

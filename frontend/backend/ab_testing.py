import random
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ABTesting:
    def __init__(self, experiment_name: str, variants: list = ["A", "B"]):
        """
        Initialize the A/B testing framework.

        Args:
            experiment_name (str): Name of the experiment.
            variants (list): List of variant labels (e.g., ["A", "B"]).
        """
        self.experiment_name = experiment_name
        self.variants = variants
        # Mapping user_id to assigned variant
        self.user_assignments = {}
        # Mapping each variant to a list of recorded metric values
        self.metrics = {variant: [] for variant in variants}

    def assign_user(self, user_id: int) -> str:
        """
        Assign a user to a variant if not already assigned.

        Args:
            user_id (int): Unique identifier for the user.

        Returns:
            str: The variant to which the user is assigned.
        """
        if user_id not in self.user_assignments:
            variant = random.choice(self.variants)
            self.user_assignments[user_id] = variant
            logger.info(f"Assigned user {user_id} to variant {variant}")
        return self.user_assignments[user_id]

    def record_metric(self, user_id: int, metric_value: float) -> None:
        """
        Record a performance metric for a user. This could be any metric such as click-through rate,
        rating score, or fairness metric.

        Args:
            user_id (int): User identifier.
            metric_value (float): Measured performance value.
        """
        variant = self.assign_user(user_id)
        self.metrics[variant].append(metric_value)
        logger.info(f"Recorded metric {metric_value:.4f} for user {user_id} in variant {variant}")

    def get_average_metric(self) -> dict:
        """
        Compute the average metric for each variant.

        Returns:
            dict: A dictionary mapping each variant to its average metric.
        """
        averages = {}
        for variant, values in self.metrics.items():
            averages[variant] = sum(values) / len(values) if values else None
        return averages

    def save_results(self, filepath: str = "ab_test_results.json") -> None:
        """
        Save the experiment results to a JSON file.

        Args:
            filepath (str): Path to the output JSON file.
        """
        results = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "user_assignments": self.user_assignments,
            "metrics": self.metrics,
            "average_metrics": self.get_average_metric()
        }
        try:
            with open(filepath, "w") as f:
                json.dump(results, f, indent=4)
            logger.info(f"Experiment results saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving experiment results: {e}")

if __name__ == "__main__":
    # Example usage: Simulate recording metrics for 10 users
    ab_tester = ABTesting("Recommendation_Experiment", variants=["A", "B"])
    for user_id in range(1, 11):
        # Simulate a metric value (e.g., a normalized performance indicator)
        metric = random.uniform(0, 1)
        ab_tester.record_metric(user_id, metric)
    averages = ab_tester.get_average_metric()
    print("Average Metrics:", averages)
    ab_tester.save_results()

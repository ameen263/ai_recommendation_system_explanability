import numpy as np
import pandas as pd
import pickle
import logging
import math
from pathlib import Path
from surprise import accuracy, Dataset, Reader
from surprise.model_selection import train_test_split
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """
    Compute Precision@K and Recall@K for the given predictions.

    Args:
        predictions (list): List of Surprise prediction objects.
        k (int): Number of top recommendations to consider.
        threshold (float): Rating threshold to consider an item as relevant.

    Returns:
        (avg_precision, avg_recall): Tuple of average precision and recall over all users.
    """
    user_est_true = defaultdict(list)
    for pred in predictions:
        user_est_true[pred.uid].append((pred.est, pred.r_ui))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value in descending order.
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        # Number of relevant items for this user.
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        # Number of recommended items in top-k that are relevant.
        n_rec_k = sum((true_r >= threshold) for (_, true_r) in user_ratings[:k])
        precisions[uid] = n_rec_k / k
        recalls[uid] = n_rec_k / n_rel if n_rel != 0 else 0
    avg_precision = np.mean(list(precisions.values()))
    avg_recall = np.mean(list(recalls.values()))
    return avg_precision, avg_recall


def ndcg_at_k(predictions, k=10, threshold=3.5):
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) at K.

    Args:
        predictions (list): List of Surprise prediction objects.
        k (int): Number of top recommendations to consider.
        threshold (float): Relevance threshold.

    Returns:
        avg_ndcg: Average NDCG over all users.
    """
    user_est_true = defaultdict(list)
    for pred in predictions:
        # Define relevance: 1 if true rating is above threshold, else 0.
        rel = 1 if pred.r_ui >= threshold else 0
        user_est_true[pred.uid].append((pred.est, rel))

    ndcgs = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort predictions by estimated value.
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        dcg = 0.0
        for i, (_, rel) in enumerate(user_ratings[:k]):
            dcg += (2 ** rel - 1) / math.log2(i + 2)
        # Compute Ideal DCG
        ideal = sorted([rel for (_, rel) in user_ratings], reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal):
            idcg += (2 ** rel - 1) / math.log2(i + 2)
        ndcgs[uid] = dcg / idcg if idcg > 0 else 0.0
    avg_ndcg = np.mean(list(ndcgs.values()))
    return avg_ndcg


class RecommenderEvaluator:
    def __init__(self,
                 data_path: str = "data/merged_data.csv",
                 model_path: str = "models/svd_model.pkl"):
        """
        Initialize the evaluator with data and model paths.

        Args:
            data_path (str): Path to the merged ratings data file.
            model_path (str): Path to the pre-trained model file.
        """
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.ratings_df = None
        self.model = None
        self.reader = Reader(rating_scale=(1, 5))  # Ratings from 1 to 5

    def load_data(self) -> bool:
        """
        Load the merged data from CSV.
        Expected format: Tab-separated file with columns: user_id, movie_id, rating, timestamp.
        """
        try:
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            df = pd.read_csv(self.data_path, sep="\t", header=0)
            df = df[["user_id", "movie_id", "rating", "timestamp"]]
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
            df.dropna(subset=["rating"], inplace=True)
            df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
            df["movie_id"] = pd.to_numeric(df["movie_id"], errors="coerce")
            df.dropna(subset=["user_id", "movie_id"], inplace=True)
            self.ratings_df = df
            logger.info(f"Successfully loaded {len(self.ratings_df)} ratings from {self.data_path}")
            return True
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise

    def load_model(self) -> bool:
        """
        Load the pre-trained collaborative filtering model from disk.
        """
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    def evaluate_model(self) -> dict:
        """
        Evaluate the recommendation model using RMSE, Precision@K, Recall@K, and NDCG.

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        try:
            self.load_data()
            self.load_model()
            data = Dataset.load_from_df(self.ratings_df[["user_id", "movie_id", "rating"]], self.reader)
            trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
            predictions = self.model.test(testset)
            if not predictions:
                raise ValueError("No predictions generated during evaluation.")

            # Compute RMSE using Surprise's accuracy module.
            rmse = accuracy.rmse(predictions, verbose=False)
            # Compute additional metrics.
            precision, recall = precision_recall_at_k(predictions, k=10, threshold=3.5)
            ndcg = ndcg_at_k(predictions, k=10, threshold=3.5)

            metrics = {
                "RMSE": round(rmse, 4),
                "Precision@K": round(precision, 4),
                "Recall@K": round(recall, 4),
                "NDCG": round(ndcg, 4)
            }
            logger.info(f"Evaluation metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
            raise


if __name__ == "__main__":
    evaluator = RecommenderEvaluator(
        data_path="data/merged_data.csv",
        model_path="models/svd_model.pkl"
    )
    try:
        results = evaluator.evaluate_model()
        print("Evaluation Results:", results)
    except Exception as err:
        print(f"Evaluation failed: {err}")

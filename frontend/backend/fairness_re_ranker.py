import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define file path for ratings data
RATINGS_FILE_PATH = "data/u.data"


def load_ratings(file_path: str = RATINGS_FILE_PATH) -> pd.DataFrame:
    """
    Load ratings data from a tab-separated file.

    Args:
        file_path (str): Path to the ratings file.

    Returns:
        pd.DataFrame: Ratings data.
    """
    try:
        ratings = pd.read_csv(file_path, sep="\t", names=["userId", "movieId", "rating", "timestamp"])
        logger.info(f"Loaded {len(ratings)} ratings from {file_path}")
        return ratings
    except Exception as e:
        logger.error(f"Error loading ratings from {file_path}: {e}")
        return pd.DataFrame()


def re_rank_fair(recommendations: list, predicted_scores: dict, alpha: float = 1.0, beta: float = 0.5) -> list:
    """
    Re-rank recommendations to mitigate fairness issues by balancing the original
    predicted relevance score with a popularity penalty.

    The adjusted score is computed as:
        adjusted_score = alpha * original_score - beta * normalized_popularity

    Args:
        recommendations (list): List of recommended movie IDs.
        predicted_scores (dict): Dictionary mapping movie IDs to their predicted scores.
        alpha (float): Weight for the original relevance score.
        beta (float): Weight for the normalized popularity penalty.

    Returns:
        list: Re-ranked list of movie IDs sorted in descending order by adjusted score.
    """
    try:
        ratings = load_ratings()
        if ratings.empty:
            logger.warning("Ratings data is empty; returning original recommendations.")
            return recommendations

        # Calculate movie popularity as the count of ratings per movie.
        movie_popularity = ratings.groupby("movieId").size()
        if movie_popularity.empty:
            logger.warning("No popularity data found; returning original recommendations.")
            return recommendations

        max_popularity = movie_popularity.max()
        adjusted_scores = {}

        for movie in recommendations:
            pop = movie_popularity.get(movie, 0)
            norm_pop = pop / max_popularity  # Normalize popularity between 0 and 1.
            original_score = predicted_scores.get(movie, 0)
            adjusted_score = alpha * original_score - beta * norm_pop
            adjusted_scores[movie] = adjusted_score
            logger.debug(
                f"Movie {movie}: original_score={original_score:.4f}, norm_pop={norm_pop:.4f}, adjusted_score={adjusted_score:.4f}")

        # Sort movie IDs by the adjusted score in descending order.
        re_ranked = sorted(adjusted_scores, key=adjusted_scores.get, reverse=True)
        logger.info(f"Re-ranked recommendations: {re_ranked}")
        return re_ranked
    except Exception as e:
        logger.error(f"Error in re-ranking recommendations: {e}")
        return recommendations


if __name__ == "__main__":
    # Example usage:
    recommended_movies = [1, 2, 3, 4, 5]
    # Simulated predicted scores for these movies.
    predicted_scores = {1: 4.5, 2: 4.2, 3: 4.0, 4: 3.8, 5: 3.5}
    re_ranked = re_rank_fair(recommended_movies, predicted_scores, alpha=1.0, beta=0.5)
    print("Re-ranked Recommendations:", re_ranked)

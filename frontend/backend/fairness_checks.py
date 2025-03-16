import pandas as pd
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define file paths (adjust as needed)
RATINGS_FILE_PATH = "data/u.data"
MOVIES_FILE_PATH = "data/u.item"  # Movies file path


def load_ratings(file_path: str) -> pd.DataFrame:
    """
    Load the ratings dataset from u.data.
    Args:
        file_path (str): Path to the ratings file.
    Returns:
        pd.DataFrame: DataFrame containing the ratings.
    """
    try:
        ratings = pd.read_csv(
            file_path,
            sep="\t",
            names=["userId", "movieId", "rating", "timestamp"]
        )
        logger.info(f"Loaded ratings data from {file_path} with {len(ratings)} records.")
        return ratings
    except Exception as e:
        logger.error(f"Error loading ratings from {file_path}: {e}")
        raise


def load_movies(file_path: str) -> pd.DataFrame:
    """
    Load the movies dataset from u.item.
    Args:
        file_path (str): Path to the movies file.
    Returns:
        pd.DataFrame: DataFrame containing the movies.
    """
    try:
        movies = pd.read_csv(
            file_path,
            sep="|",
            encoding="latin-1",
            header=None,
            usecols=[0, 1, 2],  # Select columns: movie_id, title, genres
            names=["movie_id", "title", "genres"],
        )
        logger.info(f"Loaded movies data from {file_path} with {len(movies)} records.")
        return movies
    except Exception as e:
        logger.error(f"Error loading movies from {file_path}: {e}")
        raise


def popularity_bias_score(recommendations: list) -> float:
    """
    Calculate a bias score based on movie popularity.
    Args:
        recommendations (list): List of recommended movie IDs.
    Returns:
        float: The bias score as a ratio of average popularity in recommendations to overall average popularity.
    """
    try:
        recommendations = [int(movie_id) for movie_id in recommendations]
        movie_popularity = ratings.groupby("movieId").size()
        rec_popularity = movie_popularity.reindex(recommendations).dropna()
        if rec_popularity.empty:
            logger.warning("No recommended movies found in ratings data.")
            return 0.0
        recommended_popularity = rec_popularity.mean()
        overall_popularity = movie_popularity.mean()
        if overall_popularity == 0:
            logger.warning("Overall popularity is zero; cannot compute bias score.")
            return 0.0
        bias_score = recommended_popularity / overall_popularity
        logger.info(f"Calculated popularity bias score: {bias_score:.4f}")
        return bias_score
    except Exception as e:
        logger.error(f"Error calculating popularity bias: {e}")
        return 0.0


def diversity_score(recommendations: list) -> float:
    """
    Calculate the diversity score based on unique genres per recommended movie.
    Args:
        recommendations (list): List of recommended movie IDs.
    Returns:
        float: The average number of unique genres per recommended movie.
    """
    try:
        recommendations = [int(movie_id) for movie_id in recommendations]
        recommended_movies = movies[movies["movie_id"].isin(recommendations)]
        if recommended_movies.empty:
            logger.warning("No recommended movies found in movies data.")
            return 0.0
        unique_genres_per_movie = [
            len(set(genres.split("|"))) for genres in recommended_movies["genres"] if genres
        ]
        if not unique_genres_per_movie:
            logger.warning("No genre information available for recommended movies.")
            return 0.0
        diversity = np.mean(unique_genres_per_movie)
        logger.info(f"Calculated diversity score: {diversity:.4f}")
        return diversity
    except Exception as e:
        logger.error(f"Error calculating diversity score: {e}")
        return 0.0


def exposure_fairness_score(recommendations: list) -> float:
    """
    Calculate the exposure fairness score for the given recommendations.
    Exposure fairness is measured by the coefficient of variation (std/mean) of movie popularity.
    Args:
        recommendations (list): List of recommended movie IDs.
    Returns:
        float: Exposure fairness score.
    """
    try:
        recommendations = [int(movie_id) for movie_id in recommendations]
        movie_popularity = ratings.groupby("movieId").size()
        rec_popularity = movie_popularity.reindex(recommendations).dropna()
        if rec_popularity.empty:
            logger.warning("No recommended movies found in ratings data for exposure fairness.")
            return 0.0
        mean_popularity = rec_popularity.mean()
        if mean_popularity == 0:
            logger.warning("Mean popularity of recommended movies is zero; cannot compute exposure fairness.")
            return 0.0
        exposure_fairness = rec_popularity.std() / mean_popularity
        logger.info(f"Calculated exposure fairness score: {exposure_fairness:.4f}")
        return exposure_fairness
    except Exception as e:
        logger.error(f"Error calculating exposure fairness: {e}")
        return 0.0


def check_bias_and_fairness(recommendations: list) -> dict:
    """
    Perform fairness checks for the given recommendations.

    Returns a dictionary with:
      - exposure_fairness: Computed exposure fairness score.
      - user_fairness: A placeholder value (e.g., 0.92).
      - bias_detection: A placeholder string indicating bias level.
    Args:
        recommendations (list): List of recommended movie IDs.
    Returns:
        dict: Fairness metrics.
    """
    try:
        exp_fairness = exposure_fairness_score(recommendations)
        # Placeholder value for user fairness; replace with actual computation if available.
        user_fairness = 0.92
        # Placeholder string for bias detection.
        bias_detection = "Low bias detected in collaborative filtering."

        fairness_metrics = {
            "exposure_fairness": round(exp_fairness, 4),
            "user_fairness": user_fairness,
            "bias_detection": bias_detection
        }
        logger.info(f"Fairness metrics: {fairness_metrics}")
        return fairness_metrics
    except Exception as e:
        logger.error(f"Error in fairness checks: {e}")
        return {}


# Load datasets once so they can be reused in the functions below.
try:
    ratings = load_ratings(RATINGS_FILE_PATH)
    movies = load_movies(MOVIES_FILE_PATH)
except Exception as e:
    logger.error("Failed to load datasets; please check file paths and data integrity.")

import pandas as pd
import logging
from datetime import timedelta, datetime
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SessionRecommender:
    def __init__(self, ratings_file_path: str = "data/u.data", movies_file_path: str = "data/u.item"):
        """
        Initialize the SessionRecommender with ratings and movies data.

        Args:
            ratings_file_path (str): Path to the ratings dataset.
            movies_file_path (str): Path to the movies dataset.
        """
        self.ratings_file_path = ratings_file_path
        self.movies_file_path = movies_file_path
        self.ratings_df = self._load_ratings()
        self.movies_df = self._load_movies()

    def _load_ratings(self) -> pd.DataFrame:
        """
        Load the ratings dataset.
        Expected format: tab-separated file with columns: user_id, movie_id, rating, timestamp.
        Converts Unix timestamps to datetime objects.

        Returns:
            pd.DataFrame: The ratings DataFrame.
        """
        try:
            df = pd.read_csv(
                self.ratings_file_path,
                sep="\t",
                names=["user_id", "movie_id", "rating", "timestamp"]
            )
            # Convert Unix timestamp to datetime (if not already numeric)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
            logger.info(f"Loaded {len(df)} ratings from {self.ratings_file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading ratings data: {e}")
            return pd.DataFrame()

    def _load_movies(self) -> pd.DataFrame:
        """
        Load the movies dataset.
        Expected format: pipe-separated file with columns: movie_id, title, release_date, imdb_url, genres.

        Returns:
            pd.DataFrame: The movies DataFrame.
        """
        try:
            df = pd.read_csv(
                self.movies_file_path,
                sep="|",
                encoding="latin-1",
                header=None,
                names=["movie_id", "title", "release_date", "imdb_url", "genres"]
            )
            logger.info(f"Loaded {len(df)} movies from {self.movies_file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading movies data: {e}")
            return pd.DataFrame()

    def get_context_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract context features from the provided context dictionary.

        Expected keys:
            - "timestamp": ISO formatted string (e.g., "2025-03-08T20:30:00Z")
            - "device": Device type (e.g., "mobile", "desktop")

        Returns:
            dict: Extracted features such as hour, day_of_week, and device.
        """
        features = {}
        try:
            if "timestamp" in context:
                ts = context["timestamp"].replace("Z", "+00:00")
                dt = datetime.fromisoformat(ts)
                features["hour"] = dt.hour
                features["day_of_week"] = dt.weekday()
            else:
                features["hour"] = 12
                features["day_of_week"] = 0
            features["device"] = context.get("device", "unknown").lower()
            logger.info(f"Extracted context features: {features}")
            return features
        except Exception as e:
            logger.error(f"Error extracting context features: {e}")
            return features

    def get_trending_movies(self, time_window_days: int = 30, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Identify trending movies based on the number of ratings within a given time window.

        Args:
            time_window_days (int): Number of days in the past to consider.
            top_n (int): Number of trending movies to return.

        Returns:
            List[Dict[str, Any]]: Each dictionary contains keys: movie_id, title, rating_count.
        """
        if self.ratings_df.empty or self.movies_df.empty:
            logger.warning("Ratings or movies data not available.")
            return []

        recent_threshold = pd.Timestamp.now() - timedelta(days=time_window_days)
        recent_ratings = self.ratings_df[self.ratings_df['timestamp'] >= recent_threshold]
        if recent_ratings.empty:
            logger.info("No recent ratings found in the specified time window.")
            return []

        trending_counts = recent_ratings.groupby("movie_id").size().reset_index(name="rating_count")
        trending_counts = trending_counts.sort_values("rating_count", ascending=False)
        top_trending = trending_counts.head(top_n)
        trending_movies = pd.merge(top_trending, self.movies_df[['movie_id', 'title']], on="movie_id", how="left")
        trending_list = trending_movies.to_dict(orient="records")
        logger.info(f"Identified {len(trending_list)} trending movies in the past {time_window_days} days.")
        return trending_list

    def get_session_based_recommendations(self, context: Dict[str, Any], time_window_days: int = 30, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Generate session-based recommendations by adjusting trending movie scores using session context.

        Args:
            context (dict): Contains session context like timestamp and device.
            time_window_days (int): Time window for trending analysis.
            top_n (int): Number of recommendations to return.

        Returns:
            List[Dict[str, Any]]: List of movie recommendation dictionaries.
        """
        # Get trending movies over the specified time window
        trending = self.get_trending_movies(time_window_days=time_window_days, top_n=top_n * 2)
        if not trending:
            return []

        # Extract session features
        features = self.get_context_features(context)
        hour = features.get("hour", 12)

        # Apply a boost if it's evening (e.g., 18-23)
        boost = 1.1 if 18 <= hour <= 23 else 1.0
        for movie in trending:
            movie["adjusted_score"] = movie.get("rating_count", 0) * boost

        # Sort by adjusted score and return top_n recommendations
        trending_sorted = sorted(trending, key=lambda x: x["adjusted_score"], reverse=True)
        recommendations = trending_sorted[:top_n]
        logger.info(f"Session-based recommendations generated for context: {context}")
        return recommendations

if __name__ == "__main__":
    # Example usage:
    sr = SessionRecommender()
    context = {
        "timestamp": "2025-03-08T20:30:00Z",
        "device": "mobile"
    }
    recs = sr.get_session_based_recommendations(context, time_window_days=30, top_n=5)
    print("Session-based Recommendations:")
    for rec in recs:
        print(rec)

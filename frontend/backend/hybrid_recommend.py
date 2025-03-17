import logging
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import random

from .incremental_svd import IncrementalSVD
from .content_filtering import ContentBasedRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Updated default file paths to reflect your project structure.
DEFAULT_RATINGS_FILE = Path("backend/data/merged_data.csv")
DEFAULT_MOVIES_FILE = Path("backend/data/u.item")
DEFAULT_MODEL_PATH = Path("backend/models/svd_model.pkl")


class HybridRecommender:
    def __init__(self,
                 ratings_file: Path = DEFAULT_RATINGS_FILE,
                 movies_file: Path = DEFAULT_MOVIES_FILE,
                 model_path: Path = DEFAULT_MODEL_PATH) -> None:
        """
        Initialize the HybridRecommender by loading ratings, movies, and the CF model.
        """
        self.ratings_file = ratings_file
        self.movies_file = movies_file
        self.model_path = model_path

        # Load ratings and movies; if not found, default to empty DataFrames.
        self.ratings_df: pd.DataFrame = self.load_ratings() or pd.DataFrame()
        self.movies_df: pd.DataFrame = self.load_movies() or pd.DataFrame()
        # For convenience, create an alias so tests can override movies data.
        self.movies: pd.DataFrame = self.movies_df.copy()
        self.svd_model: Optional[IncrementalSVD] = self.load_model()

        # Instantiate a content-based recommender for potential integration.
        self.content_recommender = ContentBasedRecommender(str(movies_file))

    def load_ratings(self) -> Optional[pd.DataFrame]:
        """Load ratings data from the specified file."""
        try:
            df = pd.read_csv(self.ratings_file, sep='\t')
            logger.info(f"Loaded {len(df)} ratings from {self.ratings_file}")
            return df
        except Exception as e:
            logger.error(f"Failed to load ratings data: {e}")
            return None

    def load_movies(self) -> Optional[pd.DataFrame]:
        """Load movies data from the specified file."""
        try:
            df = pd.read_csv(self.movies_file, sep='|', encoding='latin-1', header=None,
                             names=["movie_id", "title", "release_date", "imdb_url", "genres"])
            logger.info(f"Loaded {len(df)} movies from {self.movies_file}")
            return df
        except Exception as e:
            logger.error(f"Failed to load movies data: {e}")
            return None

    def load_model(self) -> Optional[IncrementalSVD]:
        """Load the collaborative filtering model from disk."""
        try:
            model = IncrementalSVD.load(str(self.model_path))
            logger.info("Collaborative filtering model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load collaborative filtering model: {e}")
            return None

    def fallback_popular_recommendations(self, top_n: int = 10) -> List[Tuple[int, str, float]]:
        """
        Return top-N popular movies based on average ratings as fallback.
        Returns:
            List of tuples: (movie_id, title, score)
        """
        logger.info("Generating fallback popular recommendations.")
        if self.ratings_df.empty:
            logger.error("Ratings data is empty; cannot compute popular recommendations.")
            return []
        popular = self.ratings_df.groupby('movie_id')['rating'].mean().sort_values(ascending=False)
        top_movie_ids = popular.head(top_n).index
        recs = self.movies_df[self.movies_df['movie_id'].isin(top_movie_ids)].copy()
        recs['score'] = recs['movie_id'].map(popular)
        return [(int(row['movie_id']), row['title'], float(row['score'])) for _, row in recs.iterrows()]

    def get_cf_recommendations(self, user_id: int) -> List[Tuple[int, float]]:
        """
        Get collaborative filtering recommendations.
        Returns a list of tuples (movie_id, cf_score) for all movies.
        """
        if self.svd_model is None or self.ratings_df.empty:
            logger.error("CF model or ratings data not available.")
            return []
        movie_ids = self.ratings_df['movie_id'].unique()
        predictions = []
        for mid in movie_ids:
            try:
                pred = self.svd_model.predict(user_id, mid).est
                predictions.append((mid, pred))
            except Exception as e:
                logger.error(f"Error predicting for movie {mid}: {e}")
        return predictions

    def hybrid_recommendation(self, user_id: int, top_n: int = 10) -> List[Tuple[int, str, float]]:
        """
        Generate hybrid recommendations by blending CF predictions with a slight diversity boost.
        Returns a list of tuples: (movie_id, title, score)
        """
        cf_recs = self.get_cf_recommendations(user_id)
        if not cf_recs:
            logger.warning("No CF recommendations; falling back to popular recommendations.")
            return self.fallback_popular_recommendations(top_n)
        cf_recs.sort(key=lambda x: x[1], reverse=True)
        top_cf = cf_recs[:top_n]
        # Use movies_df if available; otherwise, use the movies alias.
        movies_df = self.movies_df if self.movies_df is not None and not self.movies_df.empty else self.movies
        recs = []
        for mid, score in top_cf:
            movie_row = movies_df[movies_df['movie_id'] == mid]
            if movie_row.empty:
                continue
            title = movie_row.iloc[0]['title']
            adjusted_score = score + np.random.uniform(0, 0.1)
            recs.append((mid, title, adjusted_score))
        return recs

    def recommend_movies(self, user_id: int, top_n: int = 10) -> List[Tuple[int, str, float]]:
        """
        Main recommendation function that returns hybrid recommendations for a given user.
        Falls back to popular recommendations if hybrid recommendations are unavailable.
        Returns:
            List of tuples: (movie_id, title, score)
        """
        recs = self.hybrid_recommendation(user_id, top_n)
        if not recs:
            logger.warning("No hybrid recommendations available; using fallback popular recommendations.")
            recs = self.fallback_popular_recommendations(top_n)
        return recs


if __name__ == "__main__":
    test_user_id = 1
    recommender = HybridRecommender()
    recommendations = recommender.recommend_movies(test_user_id, top_n=5)
    print("Recommendations:")
    for rec in recommendations:
        print(rec)

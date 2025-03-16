import pandas as pd
import numpy as np
from surprise import SVD
import pickle
import logging
from pathlib import Path
from datetime import datetime
from content_filtering import ContentBasedRecommender

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define file paths
RATINGS_FILE_PATH = Path("data/merged_data.csv")
MOVIES_FILE_PATH = Path("data/u.item")
MODEL_PATH = Path("models/svd_model.pkl")


class HybridRecommender:
    def __init__(self, use_bert: bool = False):
        """
        Initialize the hybrid recommender system.

        Args:
            use_bert (bool): Whether to use BERT embeddings for content-based filtering.
        """
        self.ratings: pd.DataFrame = None
        self.movies: pd.DataFrame = None
        self.content_recommender: ContentBasedRecommender = None
        self.svd_model: SVD = None
        self.use_bert = use_bert
        self.initialize_system()

    def load_ratings(self) -> pd.DataFrame:
        """
        Load the ratings dataset.

        Returns:
            pd.DataFrame: DataFrame containing ratings data.
        """
        try:
            if not RATINGS_FILE_PATH.exists():
                raise FileNotFoundError(f"Ratings file not found at {RATINGS_FILE_PATH}")
            ratings = pd.read_csv(
                RATINGS_FILE_PATH,
                sep="\t",
                names=["user_id", "movie_id", "rating", "timestamp"]
            )
            ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s", errors="coerce")
            logger.info(f"Successfully loaded {len(ratings)} ratings")
            return ratings
        except Exception as e:
            logger.error(f"Error loading ratings data: {e}")
            return None

    def load_movies(self) -> pd.DataFrame:
        """
        Load the movies dataset.

        Returns:
            pd.DataFrame: DataFrame containing the movies.
        """
        try:
            if not MOVIES_FILE_PATH.exists():
                raise FileNotFoundError(f"Movies file not found at {MOVIES_FILE_PATH}")
            movies_df = pd.read_csv(
                MOVIES_FILE_PATH,
                sep="|",
                encoding="latin-1",
                header=None,
                usecols=[0, 1, 2, 4],  # movie_id, title, release_date, genres
                names=["movie_id", "title", "release_date", "genres"]
            )
            logger.info(f"Successfully loaded {len(movies_df)} movies")
            return movies_df
        except Exception as e:
            logger.error(f"Error loading movies data: {e}")
            return None

    def load_model(self) -> SVD:
        """
        Load the trained SVD model.

        Returns:
            SVD: The trained SVD model or None if not available.
        """
        try:
            if not MODEL_PATH.exists():
                logger.info(f"Model file not found at {MODEL_PATH}")
                return None
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            logger.info("Successfully loaded SVD model")
            return model
        except Exception as e:
            logger.error(f"Error loading SVD model: {e}")
            return None

    def initialize_system(self) -> None:
        """Initialize all components of the recommendation system."""
        self.ratings = self.load_ratings()
        self.movies = self.load_movies()
        self.content_recommender = ContentBasedRecommender(str(MOVIES_FILE_PATH), use_bert=self.use_bert)
        self.svd_model = self.load_model()

        if self.ratings is None or self.movies is None:
            raise RuntimeError("Failed to load ratings or movies data")

    def get_cf_recommendations(self, user_id: int, n: int = 10) -> list:
        """
        Get collaborative filtering recommendations for a user.

        Returns:
            List of tuples: (movie_id, predicted_cf_score)
        """
        if self.svd_model is None or self.movies is None:
            logger.error("Collaborative filtering model or movies data not available")
            return []
        try:
            cf_recommendations = []
            for movie_id in self.movies["movie_id"]:
                prediction = self.svd_model.predict(user_id, movie_id)
                cf_recommendations.append((movie_id, prediction.est))
            cf_recommendations.sort(key=lambda x: x[1], reverse=True)
            # Return extra candidates for later re-ranking
            return cf_recommendations[: n * 2]
        except Exception as e:
            logger.error(f"Error generating CF recommendations: {e}")
            return []

    def apply_temporal_boost(self, recommendations: list) -> list:
        """
        Apply a temporal boost to recommendations based on movie release recency.

        Returns:
            List of tuples: (movie_id, boosted_score)
        """
        boosted = []
        current_year = datetime.now().year
        for movie_id, score in recommendations:
            movie = self.content_recommender.get_movie_by_id(movie_id)
            boost = 1.0
            if movie and "release_date" in movie and movie["release_date"]:
                try:
                    # Assuming release_date ends with the year, e.g., "1995"
                    release_year = int(movie["release_date"][-4:])
                    if current_year - release_year <= 5:
                        boost = 1.1  # Boost factor for recent movies
                except Exception as e:
                    logger.warning(f"Error parsing release date for movie {movie_id}: {e}")
            boosted.append((movie_id, score * boost))
        return boosted

    def combine_with_content(self, recommendations: list) -> list:
        """
        Combine CF scores with content-based similarity.

        Returns:
            List of tuples: (movie_id, combined_score)
        """
        combined = []
        for movie_id, boosted_score in recommendations:
            content_score = 0
            similar_movies = self.content_recommender.get_similar_movies(movie_id, top_n=1)
            if similar_movies:
                content_score = similar_movies[0].get("similarity_score", 0)
            combined_score = 0.6 * boosted_score + 0.4 * content_score
            combined.append((movie_id, combined_score))
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined

    def diversity_rerank(self, recommendations: list) -> list:
        """
        Re-rank recommendations to promote diversity.

        Returns:
            List of tuples: (movie_id, title, adjusted_score)
        """
        final_recs = []
        genre_count = {}
        for movie_id, score in recommendations:
            movie = self.content_recommender.get_movie_by_id(movie_id)
            if not movie:
                continue
            genres = movie.get("genres", "")
            genre_list = [g.strip() for g in genres.split("|") if g.strip()]
            penalty = 1.0
            for genre in genre_list:
                count = genre_count.get(genre, 0)
                penalty *= 1 / (1 + count)
                genre_count[genre] = count + 1
            adjusted_score = score * penalty
            final_recs.append((movie_id, movie["title"], adjusted_score))
        final_recs.sort(key=lambda x: x[2], reverse=True)
        return final_recs

    def apply_rl_feedback(self, user_id: int, recommendations: list) -> list:
        """
        Adjust recommendations using reinforcement learning feedback.

        Returns:
            List of tuples: (movie_id, title, adjusted_score)
        """
        try:
            from rl_agent import RLAgent
            rl_agent = RLAgent()  # In production, consider reusing a persistent RL agent instance.
            recs = [(movie_id, score) for movie_id, _, score in recommendations]
            feedback = {movie_id: 0 for movie_id, _ in recs}  # Simulated neutral feedback
            updated = rl_agent.adjust_recommendations(user_id, recs, feedback)
            updated_with_title = []
            for movie_id, adj_score in updated:
                movie = self.content_recommender.get_movie_by_id(movie_id)
                title = movie.get("title", "Unknown") if movie else "Unknown"
                updated_with_title.append((movie_id, title, adj_score))
            return updated_with_title
        except Exception as e:
            logger.error(f"Error applying RL feedback: {e}")
            return recommendations

    def hybrid_recommendation(self, user_id: int, top_n: int = 10) -> list:
        """
        Generate hybrid recommendations by combining CF and content-based insights,
        applying temporal boosting, diversity re-ranking, and RL feedback.

        Returns:
            List of dictionaries with keys: item_id, title, and score.
        """
        # Adjust cold-start: if user has fewer than 2 ratings, use fallback
        user_ratings_count = self.ratings[self.ratings["user_id"] == user_id].shape[
            0] if self.ratings is not None else 0
        if user_ratings_count < 2:
            logger.info(f"User {user_id} identified as cold-start; using trending fallback recommendations.")
            trending = self.content_recommender.get_similar_movies(
                movie_id=self.movies.iloc[0]["movie_id"], top_n=top_n
            )
            return [{
                "item_id": rec.get("movie_id"),
                "title": rec.get("title", "Unknown"),
                "score": rec.get("similarity_score", 0)
            } for rec in trending]

        # Step 1: Get CF recommendations (extra candidates)
        cf_recs = self.get_cf_recommendations(user_id, n=top_n)
        # Step 2: Apply temporal boost
        boosted_recs = self.apply_temporal_boost(cf_recs)
        # Step 3: Combine CF and content-based similarity
        combined_recs = self.combine_with_content(boosted_recs)
        # Step 4: Diversity re-ranking
        diversity_recs = self.diversity_rerank(combined_recs)
        # Step 5: RL feedback adjustment
        final_recs = self.apply_rl_feedback(user_id, diversity_recs)
        # Select top_n recommendations and format output
        final_recs = final_recs[:top_n]
        output = []
        for movie_id, title, score in final_recs:
            output.append({
                "item_id": movie_id,
                "title": title,
                "score": round(score, 2)
            })
        return output


# Public function to get recommendations
def get_recommendations(user_id: int, top_n: int = 10) -> list:
    try:
        recommender = HybridRecommender(use_bert=True)  # Toggle BERT usage as needed
        return recommender.hybrid_recommendation(user_id, top_n)
    except Exception as e:
        logger.error(f"Error getting recommendations for user {user_id}: {e}")
        return []


if __name__ == "__main__":
    recs = get_recommendations(1, top_n=10)
    for rec in recs:
        print(f"Movie ID: {rec['item_id']}, Title: {rec['title']}, Score: {rec['score']:.2f}")

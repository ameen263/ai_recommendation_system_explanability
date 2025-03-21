import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import List, Optional, Dict
import logging

# Import SentenceTransformer if available
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    logging.warning("SentenceTransformer is not installed. BERT-based embeddings will not be available.")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class ContentBasedRecommender:
    def __init__(self, movies_file_path: str, use_bert: bool = False, use_tfidf: bool = True) -> None:
        """
        Initialize the content-based recommender system.

        Args:
            movies_file_path (str): Relative or absolute path to the movies data file.
            use_bert (bool): If True, use BERT-based embeddings (requires SentenceTransformer).
            use_tfidf (bool): If True, use TF-IDF vectorization; otherwise, use CountVectorizer.
        """
        self.movies_file_path = movies_file_path
        self.movies: pd.DataFrame = self._load_movies()
        self.vectorizer: Optional[object] = None
        self.count_matrix: Optional[np.ndarray] = None
        self.cosine_sim: Optional[np.ndarray] = None
        self.use_bert = use_bert and (SentenceTransformer is not None)
        self.use_tfidf = use_tfidf
        self.bert_model = None

        if self.use_bert:
            try:
                self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("BERT-based embeddings enabled using SentenceTransformer.")
            except Exception as e:
                logger.error(f"Error initializing BERT model: {e}")
                self.use_bert = False

        self._initialize_system()

    def _initialize_system(self) -> None:
        """Load movie data, create a combined text field, and compute the similarity matrix."""
        if self.movies.empty:
            logger.error("Movies dataset is empty. Check the movies file and its contents.")
            self.cosine_sim = None
            return

        if "title" not in self.movies.columns:
            logger.error("Movies dataset does not contain a 'title' column.")
            self.cosine_sim = None
            return

        # Create a combined text column using title and genres if available.
        if 'genres' in self.movies.columns:
            self.movies['combined_text'] = self.movies['title'].astype(str) + " " + self.movies['genres'].astype(str)
        else:
            self.movies['combined_text'] = self.movies['title'].astype(str)

        if self.movies['combined_text'].isnull().all() or self.movies['combined_text'].str.strip().eq("").all():
            logger.error("Combined text data is missing or empty in the dataset.")
            self.cosine_sim = None
            return

        # Compute similarity using BERT embeddings if enabled; otherwise use vectorization.
        if self.use_bert:
            self._compute_bert_embeddings()
        else:
            self._compute_similarity_matrix()

    def _load_movies(self) -> pd.DataFrame:
        """
        Load the MovieLens 'u.item' file (expected to have 24 columns) and combine genre flags.
        This method attempts multiple candidate paths if a relative path is provided.

        Returns:
            pd.DataFrame: DataFrame with columns: movie_id, title, release_date, imdb_url, genres.
        """
        candidate_paths = []
        if os.path.isabs(self.movies_file_path):
            candidate_paths.append(self.movies_file_path)
        else:
            # Candidate 1: Relative to current working directory.
            candidate_paths.append(os.path.join(os.getcwd(), self.movies_file_path))
            # Candidate 2: Relative to this file's directory.
            candidate_paths.append(os.path.join(os.path.dirname(__file__), self.movies_file_path))
            # Candidate 3: Assuming tests are run from frontend/backend/tests,
            # then "../data/u.item" should be the correct path.
            candidate_paths.append(os.path.join(os.path.dirname(__file__), "..", "data", os.path.basename(self.movies_file_path)))
            # Candidate 4: If working directory is project root ("frontend")
            candidate_paths.append(os.path.join(os.getcwd(), "backend", "data", os.path.basename(self.movies_file_path)))

        full_path = None
        for path in candidate_paths:
            if os.path.exists(path):
                full_path = os.path.abspath(path)
                break

        if full_path is None:
            logger.error(f"Movies file not found in any of the following paths: {candidate_paths}")
            return pd.DataFrame()

        try:
            col_names = [
                "movie_id", "title", "release_date", "video_release_date", "imdb_url",
                "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
                "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
                "Romance", "Sci-Fi", "Thriller", "War", "Western"
            ]
            df = pd.read_csv(
                full_path,
                sep="|",
                encoding="latin-1",
                header=None,
                names=col_names
            )

            # List of genre columns (columns 5 to 23)
            genre_cols = [
                "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
                "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
                "Romance", "Sci-Fi", "Thriller", "War", "Western"
            ]

            def combine_genres(row) -> str:
                active_genres = [g for g in genre_cols if row.get(g, 0) == 1]
                return "|".join(active_genres) if active_genres else ""

            df["genres"] = df.apply(combine_genres, axis=1)
            df = df[["movie_id", "title", "release_date", "imdb_url", "genres"]]
            logger.info(f"Loaded {len(df)} movies from {full_path}")
            return df

        except Exception as e:
            logger.error(f"Error loading movies from {full_path}: {e}")
            return pd.DataFrame()

    def _compute_similarity_matrix(self) -> None:
        """Compute the cosine similarity matrix using TF–IDF or CountVectorizer."""
        try:
            if "combined_text" not in self.movies.columns or self.movies["combined_text"].isnull().all():
                raise ValueError("Combined text data is missing or empty in the dataset.")
            if self.use_tfidf:
                self.vectorizer = TfidfVectorizer(stop_words="english")
                logger.info("Using TF-IDF vectorizer for text representation.")
            else:
                self.vectorizer = CountVectorizer(stop_words="english")
                logger.info("Using CountVectorizer for text representation.")
            self.count_matrix = self.vectorizer.fit_transform(self.movies["combined_text"])
            self.cosine_sim = cosine_similarity(self.count_matrix, self.count_matrix)
            logger.info("Cosine similarity matrix computed using text vectorization.")
        except Exception as e:
            logger.error(f"Error computing similarity matrix: {e}")
            self.cosine_sim = None

    def _compute_bert_embeddings(self) -> None:
        """Compute the cosine similarity matrix using BERT-based embeddings."""
        try:
            if "combined_text" not in self.movies.columns or self.movies["combined_text"].isnull().all():
                raise ValueError("Combined text data is missing or empty in the dataset.")
            texts = self.movies["combined_text"].tolist()
            embeddings = self.bert_model.encode(texts, show_progress_bar=True)
            self.cosine_sim = cosine_similarity(embeddings, embeddings)
            logger.info("Cosine similarity matrix computed using BERT embeddings.")
        except Exception as e:
            logger.error(f"Error computing BERT embeddings and similarity matrix: {e}")
            self.cosine_sim = None

    def get_similar_movies(self, movie_id: int, top_n: int = 10) -> List[Dict]:
        """
        Get the top_n most similar movies to the given movie_id.

        Args:
            movie_id (int): The ID of the movie to find similar movies for.
            top_n (int, optional): The number of similar movies to return.

        Returns:
            List[Dict]: A list of dictionaries with movie details and similarity score.
        """
        if self.cosine_sim is None:
            logger.error("Cosine similarity matrix not computed. Cannot process similar movies request.")
            return []
        try:
            movie_indices = self.movies.index[self.movies["movie_id"] == movie_id].tolist()
            if not movie_indices:
                logger.error(f"Movie ID {movie_id} not found in the dataset.")
                return []
            movie_idx = movie_indices[0]
            sim_scores = list(enumerate(self.cosine_sim[movie_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            # Exclude the movie itself and limit to top_n recommendations
            sim_scores = sim_scores[1:top_n + 1]

            similar_movies_list: List[Dict] = []
            for idx, sim_score in sim_scores:
                movie = self.movies.iloc[idx]
                movie_details = self._get_movie_details(movie)
                if movie_details:
                    movie_details['similarity_score'] = sim_score
                    similar_movies_list.append(movie_details)
            return similar_movies_list
        except Exception as e:
            logger.error(f"Error getting similar movies for movie_id {movie_id}: {e}")
            return []

    def _get_movie_details(self, movie: pd.Series) -> Optional[Dict]:
        """
        Extract relevant details from a movie record.

        Args:
            movie (pd.Series): A row from the movies DataFrame.

        Returns:
            Optional[Dict]: Dictionary of movie details or None on error.
        """
        try:
            return {
                "movie_id": int(movie.movie_id),
                "title": movie.title,
                "release_date": movie.release_date,
                "genres": movie.genres,
            }
        except Exception as e:
            logger.error(f"Error extracting movie details: {e}")
            return None

    def get_movie_by_id(self, movie_id: int) -> Optional[Dict]:
        """
        Retrieve details for a specific movie by its ID.

        Args:
            movie_id (int): The movie ID to lookup.

        Returns:
            Optional[Dict]: A dictionary with movie details or None if not found.
        """
        try:
            movie_rows = self.movies[self.movies["movie_id"] == movie_id]
            if movie_rows.empty:
                logger.error(f"Movie with ID {movie_id} not found.")
                return None
            movie = movie_rows.iloc[0]
            return self._get_movie_details(movie)
        except Exception as e:
            logger.error(f"Error retrieving movie with ID {movie_id}: {e}")
            return None


if __name__ == "__main__":
    try:
        # Example usage: Create a recommender using TF-IDF vectorization.
        recommender = ContentBasedRecommender("backend/data/u.item", use_bert=False, use_tfidf=True)
        example_movie_id = 1
        similar_movies = recommender.get_similar_movies(example_movie_id, top_n=5)
        if similar_movies:
            for movie in similar_movies:
                print(f"Title: {movie['title']}, Similarity Score: {movie['similarity_score']:.4f}")
        else:
            print("No similar movies found.")
    except Exception as e:
        logger.error(f"ContentBasedRecommender encountered an error: {e}")

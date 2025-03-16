import pandas as pd
import networkx as nx
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RecommendationExplainer:
    def __init__(self, movies_file_path: str = "data/u.item"):
        """
        Initialize the recommendation explainer using the standard MovieLens 100k 'u.item' format.
        """
        self.movies_file_path = movies_file_path
        self.movies_df = self._load_movies()
        self.movie_graph = self._build_movie_graph()

    def _load_movies(self) -> pd.DataFrame:
        """
        Load the standard MovieLens 100k 'u.item' file with 24 columns.

        The 24 columns are:
            0) movie_id
            1) title
            2) release_date
            3) video_release_date
            4) imdb_url
            5-23) 19 binary genre flags (0 or 1)

        This function combines columns 5 to 23 into a single 'genres' string and returns a DataFrame
        with these 5 columns: [movie_id, title, release_date, imdb_url, genres].
        """
        try:
            col_names = [
                "movie_id", "title", "release_date", "video_release_date", "imdb_url",
                "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
                "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
                "Romance", "Sci-Fi", "Thriller", "War", "Western"
            ]
            df = pd.read_csv(
                self.movies_file_path,
                sep="|",
                header=None,
                encoding="latin-1",
                names=col_names
            )
            genre_cols = [
                "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
                "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
                "Romance", "Sci-Fi", "Thriller", "War", "Western"
            ]

            def combine_genres(row):
                active_genres = [g for g in genre_cols if row[g] == 1]
                return "|".join(active_genres) if active_genres else ""

            df["genres"] = df.apply(combine_genres, axis=1)
            df = df[["movie_id", "title", "release_date", "imdb_url", "genres"]]
            logger.info(f"Loaded {len(df)} movies from {self.movies_file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading movies: {e}")
            return pd.DataFrame()

    def _build_movie_graph(self) -> nx.Graph:
        """
        Build a graph connecting movies and genres.
        Each movie node contains its title and genres, and edges connect the movie to each of its genres.
        """
        G = nx.Graph()
        try:
            for _, row in self.movies_df.iterrows():
                movie_id = row['movie_id']
                title = row['title']
                genres_str = row.get('genres', "")
                genres = [genre.strip() for genre in genres_str.split("|") if genre.strip()]
                G.add_node(movie_id, title=title, type="movie", genres=genres)
                for genre in genres:
                    if not G.has_node(genre):
                        G.add_node(genre, type="genre")
                    G.add_edge(movie_id, genre)
            logger.info("Movie graph built successfully.")
        except Exception as e:
            logger.error(f"Error building movie graph: {e}")
        return G

    def _logical_explanation(self, user_history: List[int], recommended_movie_id: int) -> str:
        """
        Generate a rule-based explanation by comparing genres of the recommended movie with movies in the user's history.
        """
        rec_movie = self.movies_df[self.movies_df['movie_id'] == recommended_movie_id]
        if rec_movie.empty:
            return "No explanation available for the recommended movie."
        rec_title = rec_movie['title'].values[0]
        rec_genres = set(rec_movie['genres'].values[0].split("|"))
        explanations = []
        if user_history:
            for hist_movie_id in user_history:
                hist_movie = self.movies_df[self.movies_df['movie_id'] == hist_movie_id]
                if hist_movie.empty:
                    continue
                hist_title = hist_movie['title'].values[0]
                hist_genres = set(hist_movie['genres'].values[0].split("|"))
                common_genres = rec_genres.intersection(hist_genres)
                if common_genres:
                    common_str = ", ".join(common_genres)
                    explanations.append(
                        f"You liked '{hist_title}', which shares the genres ({common_str}) with '{rec_title}'.")
            if explanations:
                return " ".join(explanations)
        return f"'{rec_title}' is recommended based on its unique attributes and overall popularity among similar users."

    def _graph_explanation(self, user_history: List[int], recommended_movie_id: int) -> str:
        """
        Generate a graph-based explanation using the movie graph.
        """
        if not self.movie_graph.has_node(recommended_movie_id):
            return "No explanation available for the recommended movie."
        rec_attrs = self.movie_graph.nodes[recommended_movie_id]
        rec_title = rec_attrs.get("title", "the recommended movie")
        rec_genres = set(rec_attrs.get("genres", []))
        explanation_lines = []
        for hist_movie_id in user_history:
            if not self.movie_graph.has_node(hist_movie_id):
                continue
            hist_attrs = self.movie_graph.nodes[hist_movie_id]
            hist_title = hist_attrs.get("title", "a movie")
            hist_genres = set(hist_attrs.get("genres", []))
            common_genres = rec_genres.intersection(hist_genres)
            if common_genres:
                common_str = ", ".join(common_genres)
                explanation_lines.append(
                    f"You liked '{hist_title}', which shares the genres ({common_str}) with '{rec_title}'.")
        if explanation_lines:
            return " ".join(explanation_lines)
        else:
            return f"'{rec_title}' has unique genres that might expand your viewing experience."

    def explain_recommendation(self, user_history: List[int], recommended_movie_id: int,
                               detail_level: str = "simple") -> Dict[str, Any]:
        """
        Provide an explanation for a movie recommendation using both logical and graph-based methods.

        Args:
            user_history (List[int]): List of movie IDs the user has interacted with.
            recommended_movie_id (int): The movie ID for which an explanation is needed.
            detail_level (str): "simple" returns a summary; "detailed" includes feature contributions and counterfactuals.

        Returns:
            Dict[str, Any]: If detailed, returns keys:
                - logical_explanation
                - graph_explanation
                - combined (concatenation of both)
                - feature_contributions
                - counterfactuals
            Otherwise, returns a summary explanation.
        """
        explanation_logical = self._logical_explanation(user_history, recommended_movie_id)
        explanation_graph = self._graph_explanation(user_history, recommended_movie_id)

        if detail_level.lower() == "detailed":
            # Placeholder values for demonstration; replace with dynamic computations as needed.
            feature_contributions = {
                "genre_similarity": "60%",
                "user_history": "30%",
                "popularity": "10%"
            }
            counterfactuals = ("If your rating for a similar movie had been lower, "
                               "this recommendation might not have been generated.")
            return {
                "logical_explanation": explanation_logical,
                "graph_explanation": explanation_graph,
                "combined": f"{explanation_logical} {explanation_graph}",
                "feature_contributions": feature_contributions,
                "counterfactuals": counterfactuals
            }
        else:
            return {
                "summary": explanation_logical
            }


if __name__ == "__main__":
    # Example usage:
    explainer = RecommendationExplainer()
    # Assume the user's history includes movies with IDs 1 and 2, and we want an explanation for movie ID 1.
    user_history = [1, 2]
    recommended_movie_id = 1
    simple_explanation = explainer.explain_recommendation(user_history, recommended_movie_id, detail_level="simple")
    detailed_explanation = explainer.explain_recommendation(user_history, recommended_movie_id, detail_level="detailed")

    print("Simple Explanation:")
    print(simple_explanation)
    print("\nDetailed Explanation:")
    print(detailed_explanation)

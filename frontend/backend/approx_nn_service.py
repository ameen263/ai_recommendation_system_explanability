import numpy as np
import faiss
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ApproxNNService:
    def __init__(self, embeddings: np.ndarray, movie_ids: np.ndarray, n_list: int = 100, n_probe: int = 10) -> None:
        """
        Initialize the FAISS index for approximate nearest neighbor search.

        Args:
            embeddings (np.ndarray): 2D array of shape (num_movies, embedding_dim) representing movie embeddings.
            movie_ids (np.ndarray): 1D array containing the corresponding movie IDs.
            n_list (int): The number of clusters to use (higher value -> finer partitioning).
            n_probe (int): Number of clusters to search over at query time.
        """
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")
        if movie_ids.ndim != 1:
            raise ValueError("Movie IDs must be a 1D array.")
        if embeddings.shape[0] != movie_ids.shape[0]:
            raise ValueError("Number of embeddings must match number of movie IDs.")

        self.embeddings = embeddings.astype('float32')
        self.movie_ids = movie_ids
        self.embedding_dim = self.embeddings.shape[1]
        self.n_list = n_list
        self.n_probe = n_probe
        self.index = self._build_index()

    def _build_index(self) -> faiss.Index:
        """
        Build and train a FAISS IVF index on the provided embeddings.

        Returns:
            faiss.Index: The trained FAISS index.
        """
        logger.info("Building FAISS index...")
        try:
            # Create a quantizer using a flat L2 index.
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            # Create the IVF index using L2 distance.
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.n_list, faiss.METRIC_L2)

            if not index.is_trained:
                index.train(self.embeddings)
            index.add(self.embeddings)
            index.nprobe = self.n_probe
            logger.info(f"FAISS index built and contains {index.ntotal} items.")
            return index
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            raise

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for the top_k nearest neighbors of a query embedding.

        Args:
            query_embedding (np.ndarray): A 1D array representing the query embedding.
            top_k (int): Number of nearest neighbors to return.

        Returns:
            List[Tuple[int, float]]: A list of tuples, each containing (movie_id, distance).
        """
        if query_embedding.ndim != 1:
            raise ValueError("Query embedding must be a 1D array.")
        if query_embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension {query_embedding.shape[0]} does not match expected dimension {self.embedding_dim}."
            )

        try:
            query = np.array(query_embedding, dtype='float32').reshape(1, -1)
            distances, indices = self.index.search(query, top_k)
            results: List[Tuple[int, float]] = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx != -1:  # Valid index check
                    results.append((int(self.movie_ids[idx]), float(dist)))
            return results
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []


if __name__ == "__main__":
    # Example usage:
    try:
        # Simulate movie embeddings for 100 movies, each of 64 dimensions.
        num_movies = 100
        embedding_dim = 64
        embeddings = np.random.rand(num_movies, embedding_dim).astype('float32')
        movie_ids = np.arange(1, num_movies + 1)

        # Initialize the ApproxNNService with custom FAISS parameters.
        ann_service = ApproxNNService(embeddings, movie_ids, n_list=10, n_probe=5)

        # Query with a random embedding of the same dimension.
        query_emb = np.random.rand(embedding_dim).astype('float32')
        neighbors = ann_service.search(query_emb, top_k=5)
        print("Nearest Neighbors:", neighbors)
    except Exception as e:
        logger.error(f"Error in example usage: {e}")

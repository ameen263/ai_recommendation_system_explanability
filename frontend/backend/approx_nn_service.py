import numpy as np
import faiss
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ApproxNNService:
    def __init__(self, embeddings: np.ndarray, movie_ids: np.ndarray, n_list=100, n_probe=10):
        """
        Initialize the FAISS index for approximate nearest neighbor search.

        Args:
            embeddings (np.ndarray): 2D array of shape (num_movies, embedding_dim) representing movie embeddings.
            movie_ids (np.ndarray): 1D array containing the corresponding movie IDs.
            n_list (int): The number of clusters to use (higher value -> finer partitioning).
            n_probe (int): Number of clusters to search over at query time.
        """
        self.embeddings = embeddings.astype('float32')
        self.movie_ids = movie_ids
        self.embedding_dim = self.embeddings.shape[1]
        self.n_list = n_list
        self.n_probe = n_probe
        self.index = self._build_index()

    def _build_index(self):
        """
        Build and train a FAISS index (IVF index) on the provided embeddings.

        Returns:
            index: The trained FAISS index.
        """
        logger.info("Building FAISS index...")
        # Create an IVF (inverted file) index using L2 distance
        quantizer = faiss.IndexFlatL2(self.embedding_dim)
        index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.n_list, faiss.METRIC_L2)

        # Train the index on the embeddings
        index.train(self.embeddings)
        index.add(self.embeddings)
        index.nprobe = self.n_probe
        logger.info(f"FAISS index built with {index.ntotal} items.")
        return index

    def search(self, query_embedding: np.ndarray, top_k=5):
        """
        Search for the top_k nearest neighbors of a query embedding.

        Args:
            query_embedding (np.ndarray): A 1D array representing the query embedding.
            top_k (int): Number of nearest neighbors to return.

        Returns:
            list of tuples: Each tuple contains (movie_id, distance)
        """
        query = np.array(query_embedding, dtype='float32').reshape(1, -1)
        distances, indices = self.index.search(query, top_k)
        # Map FAISS indices back to movie IDs
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:  # Ensure a valid index is returned
                results.append((int(self.movie_ids[idx]), float(dist)))
        return results


if __name__ == "__main__":
    # Example usage:
    # Simulate movie embeddings for 100 movies, each of 64 dimensions.
    num_movies = 100
    embedding_dim = 64
    embeddings = np.random.rand(num_movies, embedding_dim).astype('float32')
    movie_ids = np.arange(1, num_movies + 1)

    # Initialize the ApproxNNService
    ann_service = ApproxNNService(embeddings, movie_ids, n_list=10, n_probe=5)

    # Query with a random embedding
    query_emb = np.random.rand(embedding_dim).astype('float32')
    neighbors = ann_service.search(query_emb, top_k=5)
    print("Nearest Neighbors:", neighbors)

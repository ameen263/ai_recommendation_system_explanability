import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pandas as pd
import networkx as nx
from torch_geometric.utils import from_networkx
from sklearn.metrics.pairwise import cosine_similarity
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# In-memory cache for movie embeddings
_EMBEDDING_CACHE = {}

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def build_graph(movies_df, genre_col="genres"):
    """
    Build a bipartite graph connecting movies to their genres.
    Each movie is added as a node with type 'movie', and each genre as a node with type 'genre'.
    """
    G = nx.Graph()
    try:
        for idx, row in movies_df.iterrows():
            movie_id = row['movie_id']
            title = row['title']
            G.add_node(movie_id, type='movie', title=title)
            if isinstance(row[genre_col], str):
                genres = row[genre_col].split('|')
                for genre in genres:
                    genre = genre.strip()
                    if genre:
                        if not G.has_node(genre):
                            G.add_node(genre, type='genre')
                        G.add_edge(movie_id, genre)
        return G
    except Exception as e:
        logger.exception(f"Error building graph: {e}")
        return nx.Graph()

def prepare_graph_data(movies_df):
    """
    Prepare the graph data from the movies DataFrame.
    Creates dummy node features: For movie nodes, feature = [1.0]; for genre nodes, feature = [0.0].
    Returns the PyTorch Geometric Data object and a list of node identifiers.
    """
    G = build_graph(movies_df)
    try:
        data = from_networkx(G)
        nodes = list(G.nodes())
        features = []
        for node in nodes:
            if G.nodes[node].get("type") == "movie":
                features.append([1.0])
            else:
                features.append([0.0])
        data.x = torch.tensor(features, dtype=torch.float)
        return data, nodes
    except Exception as e:
        logger.exception(f"Error preparing graph data: {e}")
        return None, []

def train_gnn_encoder(data, epochs=100, lr=0.01, hidden_channels=16, out_channels=8):
    """
    Train a GCN encoder in an unsupervised manner using a simple edge reconstruction loss.
    Returns the trained model and the node embeddings.
    """
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GCNEncoder(in_channels=data.num_node_features, hidden_channels=hidden_channels,
                           out_channels=out_channels).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        data = data.to(device)
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            z = model(data.x, data.edge_index)
            # Unsupervised objective: maximize sigmoid(dot(z_i, z_j)) for each edge
            edge_prod = (z[data.edge_index[0]] * z[data.edge_index[1]]).sum(dim=1)
            loss = -torch.log(torch.sigmoid(edge_prod) + 1e-15).mean()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss = {loss.item():.4f}")
        model.eval()
        with torch.no_grad():
            embeddings = model(data.x, data.edge_index)
        return model, embeddings.cpu().detach()
    except Exception as e:
        logger.exception(f"Error during GNN training: {e}")
        return None, None

def get_movie_embeddings(embeddings, nodes):
    """
    Extract embeddings for movie nodes.
    Returns a dictionary mapping movie_id to its embedding.
    Implements simple in-memory caching to avoid recomputation.
    """
    global _EMBEDDING_CACHE
    if _EMBEDDING_CACHE:
        logger.info("Using cached movie embeddings.")
        return _EMBEDDING_CACHE

    movie_embeddings = {}
    try:
        for idx, node in enumerate(nodes):
            # We assume movie nodes are stored as numbers (movie IDs)
            # If the node is a string representing a genre, skip it.
            if isinstance(node, int) or (isinstance(node, str) and node.isdigit()):
                movie_embeddings[int(node)] = embeddings[idx].numpy()
        _EMBEDDING_CACHE = movie_embeddings  # Cache for future use
    except Exception as e:
        logger.exception(f"Error extracting movie embeddings: {e}")
    return movie_embeddings

def recommend_similar_movies(movie_id, movie_embeddings, top_n=5):
    """
    Recommend movies similar to the given movie_id using cosine similarity.
    Returns a list of tuples: (movie_id, similarity_score).
    """
    try:
        if movie_id not in movie_embeddings:
            logger.error(f"Movie ID {movie_id} not found in embeddings.")
            return []
        movie_embedding = movie_embeddings[movie_id].reshape(1, -1)
        all_ids = list(movie_embeddings.keys())
        all_embeddings = np.array([movie_embeddings[mid] for mid in all_ids])
        similarities = cosine_similarity(movie_embedding, all_embeddings).flatten()
        # Exclude the input movie itself and sort recommendations
        sorted_indices = np.argsort(-similarities)
        recommendations = []
        for idx in sorted_indices:
            if all_ids[idx] == movie_id:
                continue
            recommendations.append((all_ids[idx], similarities[idx]))
            if len(recommendations) >= top_n:
                break
        return recommendations
    except Exception as e:
        logger.exception(f"Error generating recommendations for movie ID {movie_id}: {e}")
        return []

if __name__ == "__main__":
    try:
        # Load movies data (assuming 'u.item' format with columns: movie_id, title, release_date, imdb_url, genres)
        movies_file = Path("data/u.item")
        movies_df = pd.read_csv(movies_file, sep="|", encoding="latin-1", header=None,
                                names=["movie_id", "title", "release_date", "imdb_url", "genres"],
                                usecols=[0, 1, 2, 4])
    except Exception as e:
        logger.error(f"Error loading movies data from {movies_file}: {e}")
        exit(1)

    data, nodes = prepare_graph_data(movies_df)
    if data is None:
        logger.error("Failed to prepare graph data. Exiting.")
        exit(1)

    model, embeddings = train_gnn_encoder(data, epochs=50, lr=0.01, hidden_channels=16, out_channels=8)
    if embeddings is None:
        logger.error("GNN training failed. Exiting.")
        exit(1)

    movie_embeddings = get_movie_embeddings(embeddings, nodes)
    # Example: Recommend movies similar to movie_id=1
    recommendations = recommend_similar_movies(1, movie_embeddings, top_n=5)
    print("Recommendations for movie_id 1:")
    for rec in recommendations:
        print(f"Movie ID: {rec[0]}, Similarity: {rec[1]:.4f}")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, mlp_layers=[64, 32, 16, 8]):
        """
        Neural Collaborative Filtering Model.

        Args:
            num_users (int): Number of unique users.
            num_items (int): Number of unique items.
            embedding_dim (int): Dimension for user/item embeddings.
            mlp_layers (list): List of layer sizes for the MLP.
        """
        super(NeuralCollaborativeFiltering, self).__init__()
        # Embedding layers for user and item
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # MLP layers; input size is 2 * embedding_dim (concatenation)
        mlp_input_size = embedding_dim * 2
        layers = []
        for layer_size in mlp_layers:
            layers.append(nn.Linear(mlp_input_size, layer_size))
            layers.append(nn.ReLU())
            mlp_input_size = layer_size
        self.mlp = nn.Sequential(*layers)

        # Final prediction layer
        self.output_layer = nn.Linear(mlp_layers[-1], 1)

    def forward(self, user_indices, item_indices):
        """
        Forward pass for the model.

        Args:
            user_indices (Tensor): Tensor of user indices.
            item_indices (Tensor): Tensor of item indices.

        Returns:
            Tensor: Predicted ratings.
        """
        user_embed = self.user_embedding(user_indices)
        item_embed = self.item_embedding(item_indices)
        # Concatenate user and item embeddings
        vector = torch.cat([user_embed, item_embed], dim=-1)
        mlp_out = self.mlp(vector)
        prediction = self.output_layer(mlp_out)
        # Output is a real-valued prediction; no activation is applied.
        return prediction.squeeze()


def train(data: pd.DataFrame, n_epochs: int = 20, lr: float = 0.005, batch_size: int = 256,
          test_size: float = 0.2, random_state: int = 42, embedding_dim: int = 32) -> (
NeuralCollaborativeFiltering, float):
    """
    Train the Neural Collaborative Filtering model on provided data.

    Args:
        data (pd.DataFrame): DataFrame with columns: user_id, movie_id, rating.
        n_epochs (int): Number of training epochs.
        lr (float): Learning rate.
        batch_size (int): Batch size for training.
        test_size (float): Proportion of data to use as test set.
        random_state (int): Random seed for splitting.
        embedding_dim (int): Dimension for embeddings.

    Returns:
        model: Trained NeuralCollaborativeFiltering model.
        rmse: RMSE on the test set.
    """
    # Ensure user and item IDs are 0-indexed integers
    data = data.copy()
    user_ids = data['user_id'].unique()
    item_ids = data['movie_id'].unique()
    user2idx = {user: idx for idx, user in enumerate(user_ids)}
    item2idx = {item: idx for idx, item in enumerate(item_ids)}
    data['user_id'] = data['user_id'].map(user2idx)
    data['movie_id'] = data['movie_id'].map(item2idx)

    num_users = len(user_ids)
    num_items = len(item_ids)
    logger.info(f"Number of users: {num_users}, Number of items: {num_items}")

    # Split data into training and testing sets
    train_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state)
    logger.info(f"Training samples: {len(train_df)}, Testing samples: {len(test_df)}")

    # Convert DataFrames to tensors
    def df_to_tensor(df):
        user_tensor = torch.tensor(df['user_id'].values, dtype=torch.long)
        item_tensor = torch.tensor(df['movie_id'].values, dtype=torch.long)
        rating_tensor = torch.tensor(df['rating'].values, dtype=torch.float)
        return user_tensor, item_tensor, rating_tensor

    train_user, train_item, train_rating = df_to_tensor(train_df)
    test_user, test_item, test_rating = df_to_tensor(test_df)

    # Initialize the model
    model = NeuralCollaborativeFiltering(num_users, num_items, embedding_dim=embedding_dim)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(n_epochs):
        permutation = torch.randperm(train_user.size()[0])
        epoch_losses = []
        for i in range(0, train_user.size()[0], batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i + batch_size]
            batch_user = train_user[indices]
            batch_item = train_item[indices]
            batch_rating = train_rating[indices]

            outputs = model(batch_user, batch_item)
            loss = criterion(outputs, batch_rating)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_loss = np.mean(epoch_losses)
        logger.info(f"Epoch {epoch + 1}/{n_epochs}: Average Loss = {avg_loss:.4f}")

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        predictions = model(test_user, test_item).cpu().numpy()
    rmse = np.sqrt(mean_squared_error(test_rating.cpu().numpy(), predictions))
    logger.info(f"Test RMSE: {rmse:.4f}")

    return model, rmse


if __name__ == "__main__":
    # Example usage:
    # Assuming you have a merged ratings file with columns: user_id, movie_id, rating, timestamp
    try:
        data = pd.read_csv("data/merged_data.csv", sep="\t")
        # If timestamp column is present, we ignore it for training
        data = data[['user_id', 'movie_id', 'rating']]
        model, test_rmse = train(data, n_epochs=20, lr=0.005, batch_size=256)
        print(f"Deep NCF Model trained. Test RMSE: {test_rmse:.4f}")
    except Exception as e:
        logger.error(f"Training failed: {e}")

#!/usr/bin/env python
"""
Train the collaborative filtering model using Surprise's SVD algorithm.
Supports incremental training using the IncrementalSVD class.
Usage:
    python train_model.py --data_path data/merged_data.csv --model_path models/svd_model.pkl
                           [--model_type svd|deep_ncf] [--incremental] [--n_factors 100] [--n_epochs 20]
                           [--lr_all 0.005] [--reg_all 0.02] [--test_size 0.2] [--random_state 42]

Options:
    --model_type   Type of model to train: 'svd' (default) or 'deep_ncf'.
    --incremental  If set, updates an existing model incrementally with new data.
    For SVD:
      --n_factors  Number of latent factors (default: 100)
      --n_epochs   Number of training epochs (default: 20)
      --lr_all     Learning rate for all parameters (default: 0.005)
      --reg_all    Regularization term for all parameters (default: 0.02)
    For deep_ncf:
      (Deep model hyperparameters can be added in the deep_ncf.py module)
"""

import os
import logging
import argparse
import pickle
from pathlib import Path
import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

# Import our IncrementalSVD for incremental training support
from incremental_svd import IncrementalSVD

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default file paths
DEFAULT_DATA_PATH = Path("data/merged_data.csv")
DEFAULT_MODEL_PATH = Path("models/svd_model.pkl")


def load_data(data_path: Path) -> pd.DataFrame:
    """
    Load merged ratings data from a CSV file.
    Expected format: Tab-separated file with columns: user_id, movie_id, rating, timestamp.

    Args:
        data_path (Path): Path to the data file.

    Returns:
        pd.DataFrame: Loaded ratings data.
    """
    if not data_path.exists():
        logger.error(f"Data file not found at {data_path}")
        raise FileNotFoundError(f"Data file not found at {data_path}")
    try:
        df = pd.read_csv(data_path, sep='\t')
        logger.info(f"Loaded {len(df)} records from {data_path}")
        return df
    except Exception as e:
        logger.exception(f"Error loading data from {data_path}: {e}")
        raise


def train_svd_model(
        data: pd.DataFrame,
        n_factors: int = 100,
        n_epochs: int = 20,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
        test_size: float = 0.2,
        random_state: int = 42
):
    """
    Train an SVD model from scratch using the Surprise library.
    Splits data into training and testing sets, trains the model, and evaluates performance using RMSE.

    Returns:
        model (SVD): The trained SVD model.
        rmse (float): RMSE on the test set.
    """
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(data[['user_id', 'movie_id', 'rating']], reader)
    trainset, testset = train_test_split(dataset, test_size=test_size, random_state=random_state)
    logger.info("Data split into training and testing sets.")

    model = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
        random_state=random_state
    )
    model.fit(trainset)
    logger.info("SVD model training completed from scratch.")

    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    logger.info(f"Model evaluation completed: RMSE = {rmse:.4f}")
    return model, rmse


def main():
    parser = argparse.ArgumentParser(description="Train the movie recommendation model.")
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATA_PATH),
                        help="Path to the merged ratings data CSV file")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH),
                        help="Path to save or load the trained model")
    parser.add_argument("--model_type", type=str, default="svd", choices=["svd", "deep_ncf"],
                        help="Type of model to train: 'svd' (default) or 'deep_ncf'")
    parser.add_argument("--incremental", action="store_true",
                        help="If set, update an existing SVD model incrementally with new data")
    # SVD-specific hyperparameters
    parser.add_argument("--n_factors", type=int, default=100, help="Number of latent factors (SVD)")
    parser.add_argument("--n_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr_all", type=float, default=0.005, help="Learning rate for all parameters")
    parser.add_argument("--reg_all", type=float, default=0.02, help="Regularization term for all parameters")
    # Common hyperparameters
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of data for test set")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for data splitting")

    args = parser.parse_args()

    try:
        data = load_data(Path(args.data_path))
        # If incremental training is enabled and model file exists, load and update the model.
        if args.model_type == "svd":
            if args.incremental and os.path.exists(args.model_path):
                # Load existing incremental model
                logger.info("Incremental flag set and model exists. Loading existing IncrementalSVD model for update.")
                model = IncrementalSVD.load(args.model_path)
                # Use partial_fit to update the model with new data.
                model.partial_fit(data)
                # For evaluation, we can compute RMSE on a held-out subset.
                # Here, we skip detailed evaluation in incremental update for brevity.
                rmse = None
                logger.info("Incremental training completed.")
            else:
                # Train SVD model from scratch using our standard approach.
                logger.info("Training SVD model from scratch.")
                model_obj, rmse = train_svd_model(
                    data,
                    n_factors=args.n_factors,
                    n_epochs=args.n_epochs,
                    lr_all=args.lr_all,
                    reg_all=args.reg_all,
                    test_size=args.test_size,
                    random_state=args.random_state
                )
                # Wrap the SVD model into an IncrementalSVD instance for consistency.
                model = IncrementalSVD(n_factors=args.n_factors, n_epochs=args.n_epochs,
                                       lr_all=args.lr_all, reg_all=args.reg_all, random_state=args.random_state)
                # We store the training data inside our incremental model.
                model.fit(data)
        else:
            # If using deep_ncf, you would call the deep_ncf module's train function.
            # For this example, we assume deep_ncf training is handled separately.
            raise NotImplementedError("Deep NCF training is not implemented in this example.")

        # Save the (incremental) model to disk
        with open(args.model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved successfully at {args.model_path}")

        if rmse is not None:
            logger.info(f"Training completed with RMSE: {rmse:.4f}")
            print(f"Training completed with RMSE: {rmse:.4f}")
        else:
            logger.info("Incremental training completed without RMSE evaluation.")
            print("Incremental training completed.")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"Training failed: {e}")


if __name__ == "__main__":
    main()

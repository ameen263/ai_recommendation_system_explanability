#!/usr/bin/env python
"""
Train the collaborative filtering model using Surprise's SVD algorithm.
Supports incremental training using the IncrementalSVD class.

Usage:
    python train_model.py --data_path data/merged_data.csv --model_path models/svd_model.pkl
                           [--incremental] [--n_factors 100] [--n_epochs 20]
                           [--lr_all 0.005] [--reg_all 0.02] [--test_size 0.2] [--random_state 42]
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import pandas as pd
from surprise import Dataset, Reader
# Use a relative import to correctly reference the IncrementalSVD class in the same package.
from .incremental_svd import IncrementalSVD

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path("data/merged_data.csv")
DEFAULT_MODEL_PATH = Path("models/svd_model.pkl")


def load_data(data_path) -> pd.DataFrame:
    # Convert to Path if a string is provided.
    if not isinstance(data_path, Path):
        data_path = Path(data_path)
    if not data_path.exists():
        logger.error(f"Data file not found at {data_path}")
        sys.exit(1)  # Exit with code 1 as expected by tests
    try:
        # Load data assuming tab-separated values
        df = pd.read_csv(data_path, sep='\t')
        logger.info(f"Loaded {len(df)} records from {data_path}")
        # Normalize column names: rename camelCase to snake_case if needed
        if 'userId' in df.columns:
            df.rename(columns={'userId': 'user_id'}, inplace=True)
        if 'movieId' in df.columns:
            df.rename(columns={'movieId': 'movie_id'}, inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Train the collaborative filtering model.")
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATA_PATH),
                        help="Path to merged ratings data CSV")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH),
                        help="Path to save the trained model")
    parser.add_argument("--incremental", action="store_true", help="Incrementally update model")
    parser.add_argument("--n_factors", type=int, default=100)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--lr_all", type=float, default=0.005)
    parser.add_argument("--reg_all", type=float, default=0.02)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()

    data_path = Path(args.data_path)
    model_path = Path(args.model_path)

    data = load_data(data_path)

    if args.incremental and model_path.exists():
        logger.info("Loading existing IncrementalSVD model for incremental update.")
        try:
            model = IncrementalSVD.load(str(model_path))
            model.partial_fit(data)
            logger.info("Incremental update completed successfully.")
        except Exception as e:
            logger.error("Error during incremental update: %s", e)
            sys.exit(1)
    else:
        logger.info("Training a new IncrementalSVD model from scratch.")
        model = IncrementalSVD(n_factors=args.n_factors, n_epochs=args.n_epochs,
                               lr_all=args.lr_all, reg_all=args.reg_all,
                               random_state=args.random_state)
        model.fit(data)
        logger.info("Model trained from scratch.")

    # Ensure the directory for saving the model exists
    model_dir = model_path.parent
    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)

    try:
        model.save(str(model_path))
        logger.info(f"Model saved successfully at {model_path}")
    except Exception as e:
        logger.error("Error saving the model: %s", e)
        sys.exit(1)


if __name__ == '__main__':
    main()

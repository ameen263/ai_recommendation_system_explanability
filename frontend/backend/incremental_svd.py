import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader
import pickle
import logging

# Configure logging for the module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class IncrementalSVD:
    """
    A wrapper class for the Surprise SVD algorithm that simulates incremental training.
    This implementation retrains on the full dataset after appending new data.
    """

    def __init__(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.random_state = random_state
        self.model = SVD(n_factors=self.n_factors,
                         n_epochs=self.n_epochs,
                         lr_all=self.lr_all,
                         reg_all=self.reg_all,
                         random_state=self.random_state)
        self.train_data = None  # Holds cumulative training data

    def fit(self, ratings_df: pd.DataFrame):
        """
        Train the SVD model on the initial ratings dataset.

        Parameters:
            ratings_df (pd.DataFrame): DataFrame with columns ['user_id', 'movie_id', 'rating']
        """
        try:
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)
            trainset = data.build_full_trainset()
            self.train_data = ratings_df.copy()  # Store training data for future incremental updates
            self.model.fit(trainset)
            logger.info("IncrementalSVD model trained on initial dataset with %d records.", len(ratings_df))
        except Exception as e:
            logger.error("Error during initial training: %s", e)
            raise

    def partial_fit(self, new_ratings_df: pd.DataFrame):
        """
        Update the model with new ratings by appending the new data to the cumulative training set,
        then retraining the model on the full dataset.

        Parameters:
            new_ratings_df (pd.DataFrame): New ratings with columns ['user_id', 'movie_id', 'rating']
        """
        try:
            if self.train_data is None:
                # If there's no previous training data, perform initial training.
                logger.info("No existing training data. Fitting on new data.")
                self.fit(new_ratings_df)
                return

            # Append new ratings to the existing training data.
            self.train_data = pd.concat([self.train_data, new_ratings_df], ignore_index=True)
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(self.train_data[['user_id', 'movie_id', 'rating']], reader)
            trainset = data.build_full_trainset()

            # Adjust the number of epochs for incremental update (warm start).
            additional_epochs = 5  # You may tune this value based on your needs.
            self.model.n_epochs = additional_epochs
            self.model.fit(trainset)
            logger.info("IncrementalSVD model updated with %d new records; total records: %d.",
                        len(new_ratings_df), len(self.train_data))
        except Exception as e:
            logger.error("Error during incremental update: %s", e)
            raise

    def predict(self, user_id, movie_id):
        """
        Predict the rating for a given user and movie.

        Parameters:
            user_id: Identifier for the user.
            movie_id: Identifier for the movie.

        Returns:
            Prediction object from Surprise (with .est attribute for the estimated rating)
        """
        try:
            prediction = self.model.predict(user_id, movie_id)
            logger.info("Predicted rating for user %s, movie %s: %.4f", user_id, movie_id, prediction.est)
            return prediction
        except Exception as e:
            logger.error("Error during prediction: %s", e)
            raise

    def save(self, file_path: str):
        """
        Save the entire IncrementalSVD object to disk using pickle.

        Parameters:
            file_path (str): Path to save the model.
        """
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)
            logger.info("IncrementalSVD model saved to %s", file_path)
        except Exception as e:
            logger.error("Error saving the model: %s", e)
            raise

    @staticmethod
    def load(file_path: str):
        """
        Load an IncrementalSVD object from disk.

        Parameters:
            file_path (str): Path from which to load the model.

        Returns:
            An instance of IncrementalSVD.
        """
        try:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            logger.info("IncrementalSVD model loaded from %s", file_path)
            return model
        except Exception as e:
            logger.error("Error loading the model from %s: %s", file_path, e)
            raise


if __name__ == "__main__":
    # Example usage demonstrating initial training and incremental update.

    # Create an initial dataset.
    initial_data = pd.DataFrame({
        'user_id': [1, 2, 1],
        'movie_id': [10, 20, 30],
        'rating': [4.0, 3.5, 5.0]
    })

    # Simulate new incoming ratings.
    new_data = pd.DataFrame({
        'user_id': [2, 3],
        'movie_id': [30, 10],
        'rating': [4.0, 3.0]
    })

    # Initialize and train the model on the initial dataset.
    inc_svd = IncrementalSVD(n_factors=50, n_epochs=20)
    inc_svd.fit(initial_data)
    print("Initial prediction for user 1, movie 10:", inc_svd.predict(1, 10).est)

    # Update the model incrementally with new data.
    inc_svd.partial_fit(new_data)
    print("Updated prediction for user 1, movie 10:", inc_svd.predict(1, 10).est)

    # Save the model.
    inc_svd.save("incremental_svd_model.pkl")

    # Load the model and test prediction.
    loaded_model = IncrementalSVD.load("incremental_svd_model.pkl")
    print("Loaded model prediction for user 1, movie 10:", loaded_model.predict(1, 10).est)

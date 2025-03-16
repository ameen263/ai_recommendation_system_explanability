import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader
import pickle
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class IncrementalSVD:
    def __init__(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42):
        """
        Initialize the IncrementalSVD model with hyperparameters.
        """
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
        self.train_data = None  # To store the cumulative training data

    def fit(self, ratings_df):
        """
        Fit the SVD model on the initial ratings dataframe.
        Expected ratings_df columns: ['user_id', 'movie_id', 'rating']
        """
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)
        trainset = data.build_full_trainset()
        self.train_data = ratings_df.copy()
        self.model.fit(trainset)
        logger.info("IncrementalSVD model trained on initial dataset.")

    def partial_fit(self, new_ratings_df):
        """
        Incrementally update the model using new ratings.
        This appends new data to the existing training set and performs additional training.
        """
        if self.train_data is None:
            # If no model exists yet, train on new data
            logger.info("No existing training data. Fitting on new data.")
            self.fit(new_ratings_df)
            return

        # Append new ratings to the cumulative training data
        self.train_data = pd.concat([self.train_data, new_ratings_df], ignore_index=True)
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.train_data[['user_id', 'movie_id', 'rating']], reader)
        trainset = data.build_full_trainset()

        # Use a few additional epochs for incremental updates (warm start)
        additional_epochs = 5  # You may adjust this based on your needs
        self.model.n_epochs = additional_epochs
        self.model.fit(trainset)
        logger.info("IncrementalSVD model updated with new ratings.")

    def predict(self, user_id, movie_id):
        """
        Predict the rating for a given user and movie.
        """
        return self.model.predict(user_id, movie_id)

    def save(self, file_path):
        """
        Save the entire IncrementalSVD object (including training data and model parameters) to disk.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"IncrementalSVD model saved to {file_path}")

    @staticmethod
    def load(file_path):
        """
        Load an IncrementalSVD object from disk.
        """
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"IncrementalSVD model loaded from {file_path}")
        return model

if __name__ == "__main__":
    # Example usage
    # Initial training data
    initial_data = pd.DataFrame({
        'user_id': [1, 2, 1],
        'movie_id': [10, 20, 30],
        'rating': [4.0, 3.5, 5.0]
    })

    # New ratings arriving later
    new_data = pd.DataFrame({
        'user_id': [2, 3],
        'movie_id': [30, 10],
        'rating': [4.0, 3.0]
    })

    # Create and train the model initially
    inc_svd = IncrementalSVD(n_factors=50, n_epochs=20)
    inc_svd.fit(initial_data)
    print("Initial prediction for user 1, movie 10:", inc_svd.predict(1, 10).est)

    # Incrementally update the model with new data
    inc_svd.partial_fit(new_data)
    print("Updated prediction for user 1, movie 10:", inc_svd.predict(1, 10).est)

    # Save and load the model
    inc_svd.save("incremental_svd_model.pkl")
    loaded_model = IncrementalSVD.load("incremental_svd_model.pkl")
    print("Loaded model prediction for user 1, movie 10:", loaded_model.predict(1, 10).est)

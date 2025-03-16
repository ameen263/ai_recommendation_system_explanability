import pandas as pd
import sqlite3
import logging

# Paths to your datasets
RATINGS_DATA_PATH = 'data/u.data'
USERS_DATA_PATH = 'data/u.user'
MERGED_DATA_PATH = 'data/merged_data.csv'  # Output path for merged data

# Configure logging
logging.basicConfig(
    filename='logs/integrate_data.log',  # Log file name
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_users(file_path: str) -> pd.DataFrame:
    """
    Loads user demographic data from the u.user file.

    Expected file format (pipe-separated):
      user_id | age | gender | occupation | zip_code

    :param file_path: Path to the u.user file.
    :return: DataFrame containing user demographic data.
    """
    try:
        users_df = pd.read_csv(
            file_path,
            sep="|",
            header=None,
            names=["user_id", "age", "gender", "occupation", "zip_code"]
        )
        logging.info("Successfully loaded users from %s", file_path)
        return users_df
    except Exception as e:
        logging.error("Error loading users from %s: %s", file_path, e)
        return None


def load_ratings(data_path: str) -> pd.DataFrame:
    """
    Loads and preprocesses the ratings dataset.

    Expected file format (tab-separated):
      user_id  movie_id  rating  timestamp

    :param data_path: Path to the ratings data file.
    :return: DataFrame containing ratings data.
    """
    try:
        ratings_df = pd.read_csv(
            data_path,
            sep='\t',
            header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )
        logging.info("Successfully loaded ratings data from %s", data_path)
        return ratings_df
    except Exception as e:
        logging.error("Error loading ratings data from %s: %s", data_path, e)
        return None


def merge_datasets(users_df: pd.DataFrame, ratings_df: pd.DataFrame) -> (pd.DataFrame, dict):
    """
    Merges user demographic data with ratings data.

    The merge is performed on the 'user_id' field.

    :param users_df: DataFrame containing user demographic data.
    :param ratings_df: DataFrame containing ratings data.
    :return: Tuple of (merged DataFrame, user ID mapping dictionary)
             or (None, None) on error.
    """
    try:
        # Merge ratings with users data on user_id using an inner join to drop orphaned ratings
        merged_df = pd.merge(
            ratings_df,
            users_df,
            on="user_id",
            how="inner"
        )

        dropped = len(ratings_df) - len(merged_df)
        if dropped > 0:
            logging.info("Dropped %d rating records with no matching user demographic data.", dropped)

        # Deduplicate the merged data
        merged_df.drop_duplicates(inplace=True)
        logging.info("Successfully merged datasets with %d records after deduplication.", len(merged_df))

        # Build a mapping from new user IDs to themselves (or optionally from old to new)
        user_id_map = merged_df.set_index('user_id')['user_id'].to_dict()

        return merged_df, user_id_map
    except Exception as e:
        logging.error("Error merging datasets: %s", e)
        return None, None


if __name__ == "__main__":
    users_df = load_users(USERS_DATA_PATH)
    ratings_df = load_ratings(RATINGS_DATA_PATH)

    if users_df is not None and ratings_df is not None:
        merged_df, user_map = merge_datasets(users_df, ratings_df)
        if merged_df is not None:
            merged_df.to_csv(MERGED_DATA_PATH, sep='\t', index=False)
            logging.info("Merged dataset saved to %s", MERGED_DATA_PATH)


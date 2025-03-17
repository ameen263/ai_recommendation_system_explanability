import pandas as pd
import sqlite3
import logging
import os

# Define paths to datasets
RATINGS_DATA_PATH = 'data/u.data'
USERS_DATA_PATH = 'data/u.user'
MERGED_DATA_PATH = 'data/merged_data.csv'  # Output path for merged data

# Ensure the logs directory exists
LOG_DIR = 'logs'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'integrate_data.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_users(file_path: str) -> pd.DataFrame:
    """
    Loads user demographic data from the u.user file.

    Expected file formats:
      - Standard: user_id | age | gender | occupation | zip_code
      - Alternative: id | username

    If columns differ, they are renamed to use 'user_id' consistently.

    :param file_path: Path to the u.user file.
    :return: DataFrame containing user demographic data.
    """
    try:
        users_df = pd.read_csv(file_path, sep="|", header=None)
        # Determine format based on number of columns
        if users_df.shape[1] == 5:
            users_df.columns = ["user_id", "age", "gender", "occupation", "zip_code"]
        elif users_df.shape[1] == 2:
            users_df.columns = ["id", "username"]
        else:
            logging.warning("Unexpected number of columns (%d) in users file.", users_df.shape[1])

        # Rename 'id' to 'user_id' if necessary
        if 'id' in users_df.columns and 'user_id' not in users_df.columns:
            users_df.rename(columns={'id': 'user_id'}, inplace=True)

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

    If a different column name is used (e.g., 'user_id_old'), it is renamed.

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
        # Handle alternative column name if necessary
        if 'user_id' not in ratings_df.columns and 'user_id_old' in ratings_df.columns:
            ratings_df.rename(columns={'user_id_old': 'user_id'}, inplace=True)
        logging.info("Successfully loaded ratings data from %s", data_path)
        return ratings_df
    except Exception as e:
        logging.error("Error loading ratings data from %s: %s", data_path, e)
        return None


def merge_datasets(users_df: pd.DataFrame, ratings_df: pd.DataFrame) -> (pd.DataFrame, dict):
    """
    Merges user demographic data with ratings data on the 'user_id' field.
    Handles column name mismatches by renaming where necessary.
    Deduplicates records and builds a simple user ID mapping.

    :param users_df: DataFrame containing user demographic data.
    :param ratings_df: DataFrame containing ratings data.
    :return: Tuple of (merged DataFrame, user ID mapping dictionary)
             or (None, None) on error.
    """
    try:
        # Ensure key columns are correctly named
        if 'user_id' not in ratings_df.columns and 'user_id_old' in ratings_df.columns:
            ratings_df.rename(columns={'user_id_old': 'user_id'}, inplace=True)
        if 'user_id' not in users_df.columns and 'id' in users_df.columns:
            users_df.rename(columns={'id': 'user_id'}, inplace=True)

        # Merge ratings with users using an inner join (dropping orphaned ratings)
        merged_df = pd.merge(ratings_df, users_df, on="user_id", how="inner")

        dropped = len(ratings_df) - len(merged_df)
        if dropped > 0:
            logging.info("Dropped %d rating records with no matching user demographic data.", dropped)

        # Deduplicate merged data
        merged_df.drop_duplicates(inplace=True)
        logging.info("Successfully merged datasets with %d records after deduplication.", len(merged_df))

        # Build a simple mapping from user_id to itself (extendable if needed)
        user_id_map = merged_df['user_id'].to_dict()

        return merged_df, user_id_map
    except Exception as e:
        logging.error("Error merging datasets: %s", e)
        return None, None


if __name__ == "__main__":
    # Ensure the output directory for merged data exists
    data_dir = os.path.dirname(MERGED_DATA_PATH)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    users_df = load_users(USERS_DATA_PATH)
    ratings_df = load_ratings(RATINGS_DATA_PATH)

    if users_df is not None and ratings_df is not None:
        merged_df, user_map = merge_datasets(users_df, ratings_df)
        if merged_df is not None:
            merged_df.to_csv(MERGED_DATA_PATH, sep='\t', index=False)
            logging.info("Merged dataset saved to %s", MERGED_DATA_PATH)
        else:
            logging.error("Merging datasets failed.")
    else:
        logging.error("Failed to load users or ratings data.")

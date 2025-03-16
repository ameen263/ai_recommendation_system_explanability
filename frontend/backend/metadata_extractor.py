import pandas as pd
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MetadataExtractor:
    def __init__(self, metadata_file_path: str = "data/movie_metadata.csv"):
        """
        Initialize the MetadataExtractor.

        Args:
            metadata_file_path (str): Path to the movie metadata CSV file.
                Expected columns: movie_id, title, plot_summary, cast, director.
        """
        self.metadata_file_path = metadata_file_path
        self.metadata_df: Optional[pd.DataFrame] = None

    def load_metadata(self) -> pd.DataFrame:
        """
        Load and return the metadata DataFrame from the CSV file.

        Returns:
            pd.DataFrame: DataFrame with metadata.
        """
        try:
            if not os.path.exists(self.metadata_file_path):
                raise FileNotFoundError(f"Metadata file not found at {self.metadata_file_path}")
            self.metadata_df = pd.read_csv(self.metadata_file_path)
            logger.info(f"Loaded metadata for {len(self.metadata_df)} movies from {self.metadata_file_path}")
            return self.metadata_df
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            raise

    def preprocess_metadata(self) -> pd.DataFrame:
        """
        Preprocess the metadata by combining key textual fields into a single column.

        The method expects columns: 'movie_id', 'title', 'plot_summary', 'cast', 'director'.
        It fills missing values with an empty string and concatenates the fields.

        Returns:
            pd.DataFrame: DataFrame with an additional 'combined_text' column.
        """
        try:
            if self.metadata_df is None:
                self.load_metadata()

            # Define the columns to combine; adjust as needed.
            text_columns = ['title', 'plot_summary', 'cast', 'director']
            for col in text_columns:
                if col not in self.metadata_df.columns:
                    logger.warning(f"Column '{col}' not found in metadata. Filling with empty strings.")
                    self.metadata_df[col] = ""
                else:
                    # Fill missing values with empty strings
                    self.metadata_df[col] = self.metadata_df[col].fillna("")

            # Create a combined text column
            self.metadata_df['combined_text'] = (
                    self.metadata_df['title'] + " " +
                    self.metadata_df['plot_summary'] + " " +
                    self.metadata_df['cast'] + " " +
                    self.metadata_df['director']
            ).str.strip()

            logger.info("Metadata preprocessing complete. 'combined_text' column created.")
            return self.metadata_df
        except Exception as e:
            logger.error(f"Error preprocessing metadata: {e}")
            raise


if __name__ == "__main__":
    # Example usage:
    try:
        extractor = MetadataExtractor("data/movie_metadata.csv")
        metadata = extractor.load_metadata()
        processed_metadata = extractor.preprocess_metadata()
        # Display first few rows of the processed DataFrame
        print(processed_metadata.head())
    except Exception as e:
        logger.error(f"Metadata extraction failed: {e}")

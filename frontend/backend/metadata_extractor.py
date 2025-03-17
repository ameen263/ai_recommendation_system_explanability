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
        Raises:
            FileNotFoundError: If the metadata file is not found.
            Exception: For other errors during file reading.
        """
        if not os.path.exists(self.metadata_file_path):
            msg = f"Metadata file not found at {self.metadata_file_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        try:
            # Use UTF-8 encoding by default; adjust if needed
            self.metadata_df = pd.read_csv(self.metadata_file_path, encoding="utf-8")
            logger.info(f"Loaded metadata for {len(self.metadata_df)} movies from {self.metadata_file_path}")
            return self.metadata_df
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            raise

    def preprocess_metadata(self) -> pd.DataFrame:
        """
        Preprocess the metadata by combining key textual fields into a single column.

        Expected columns: 'movie_id', 'title', 'plot_summary', 'cast', 'director'.
        This method fills missing values with an empty string and concatenates these fields into
        a new column 'combined_text'.

        Returns:
            pd.DataFrame: DataFrame with an additional 'combined_text' column.
        Raises:
            Exception: If preprocessing fails.
        """
        try:
            if self.metadata_df is None:
                self.load_metadata()

            # Define the columns to combine
            text_columns = ['title', 'plot_summary', 'cast', 'director']
            for col in text_columns:
                if col not in self.metadata_df.columns:
                    logger.warning(f"Column '{col}' not found in metadata. Creating column with empty strings.")
                    self.metadata_df[col] = ""
                else:
                    # Fill any missing values with empty strings
                    self.metadata_df[col] = self.metadata_df[col].fillna("")

            # Create a combined text column by joining the specified columns
            self.metadata_df['combined_text'] = self.metadata_df[text_columns]\
                .apply(lambda row: " ".join(row.astype(str)).strip(), axis=1)

            logger.info("Metadata preprocessing complete. 'combined_text' column created successfully.")
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

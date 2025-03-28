import numpy as np
import logging
import sqlite3
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define the SQLite database path for persistent Q-values
DATABASE_PATH = Path("rl_agent.db")

class RLAgent:
    def __init__(self, learning_rate: float = 0.01, discount_factor: float = 0.9, db_path: Path = DATABASE_PATH) -> None:
        """
        Initialize the RL agent with specified learning parameters and persistent storage.

        Args:
            learning_rate (float): The rate at which the agent learns from feedback.
            discount_factor (float): The discount factor for future rewards.
            db_path (Path): Path to the SQLite database for persistent Q-values.
        """
        self.learning_rate: float = learning_rate
        self.discount_factor: float = discount_factor
        self.db_path: Path = db_path
        try:
            self.conn: sqlite3.Connection = sqlite3.connect(str(self.db_path))
            self.conn.row_factory = sqlite3.Row
        except Exception as e:
            logger.error(f"Error connecting to database at {db_path}: {e}")
            raise
        self._create_table()
        self.q_values: Dict[Tuple[int, int], float] = self._load_q_values()

    def _create_table(self) -> None:
        """
        Create the table for storing Q-values if it does not exist.
        """
        try:
            with self.conn:
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS rl_qvalues (
                        user_id INTEGER,
                        movie_id INTEGER,
                        q_value REAL,
                        PRIMARY KEY (user_id, movie_id)
                    )
                """)
            logger.info("RL Q-values table ensured in the database.")
        except Exception as e:
            logger.error(f"Error creating RL Q-values table: {e}")
            raise

    def _load_q_values(self) -> Dict[Tuple[int, int], float]:
        """
        Load Q-values from persistent storage.

        Returns:
            dict: Dictionary with keys as (user_id, movie_id) and values as q_value.
        """
        q_vals: Dict[Tuple[int, int], float] = {}
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT user_id, movie_id, q_value FROM rl_qvalues")
            rows = cursor.fetchall()
            for row in rows:
                q_vals[(row["user_id"], row["movie_id"])] = row["q_value"]
            logger.info(f"Loaded {len(q_vals)} Q-values from the database.")
        except Exception as e:
            logger.error(f"Error loading Q-values from database: {e}")
        return q_vals

    def _save_q_value(self, user_id: int, movie_id: int, q_value: float) -> None:
        """
        Save or update a Q-value in persistent storage.

        Args:
            user_id (int): Identifier of the user.
            movie_id (int): Identifier of the movie.
            q_value (float): The Q-value to save.
        """
        try:
            with self.conn:
                self.conn.execute("""
                    INSERT INTO rl_qvalues (user_id, movie_id, q_value)
                    VALUES (?, ?, ?)
                    ON CONFLICT(user_id, movie_id) DO UPDATE SET q_value=excluded.q_value
                """, (user_id, movie_id, q_value))
            logger.info(f"Persisted Q-value for user {user_id}, movie {movie_id}: {q_value:.4f}")
        except Exception as e:
            logger.error(f"Error saving Q-value for user {user_id}, movie {movie_id}: {e}")

    def get_q_value(self, user_id: int, movie_id: int) -> float:
        """
        Retrieve the current Q-value for a given user and movie.

        Args:
            user_id (int): User identifier.
            movie_id (int): Movie identifier.

        Returns:
            float: The current Q-value (defaults to 0.0 if not present).
        """
        return self.q_values.get((user_id, movie_id), 0.0)

    def update_q_value(self, user_id: int, movie_id: int, reward: float, next_max: float = 0.0) -> float:
        """
        Update the Q-value for a specific user and movie based on the received reward.
        The update is computed using the formula:
            Q(s,a) <- Q(s,a) + learning_rate * (reward + discount_factor * next_max - Q(s,a))

        Args:
            user_id (int): User identifier.
            movie_id (int): Movie identifier.
            reward (float): Reward received (e.g., +1 for positive feedback, -1 for negative).
            next_max (float): The maximum Q-value for subsequent states (default is 0.0).

        Returns:
            float: The updated Q-value.
        """
        current_q = self.get_q_value(user_id, movie_id)
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max - current_q)
        self.q_values[(user_id, movie_id)] = new_q
        self._save_q_value(user_id, movie_id, new_q)
        logger.info(f"Updated Q-value for user {user_id}, movie {movie_id}: {new_q:.4f}")
        return new_q

    def adjust_recommendations(self, user_id: int, recommendations: list, feedback: dict) -> list:
        """
        Adjust recommendation scores based on reinforcement learning feedback.
        For each recommended movie, update its Q-value and blend it with the original score.

        Args:
            user_id (int): The user's identifier.
            recommendations (list): List of tuples (movie_id, original_score).
            feedback (dict): Mapping of movie_id to feedback reward.

        Returns:
            list: Updated recommendations as tuples (movie_id, adjusted_score), sorted descending by score.
        """
        updated_recommendations = []
        # Calculate the maximum current Q-value among recommended movies.
        next_max = max([self.get_q_value(user_id, movie_id) for movie_id, _ in recommendations], default=0)
        for movie_id, score in recommendations:
            reward = feedback.get(movie_id, 0)
            new_q = self.update_q_value(user_id, movie_id, reward, next_max)
            # Blend original score (70%) with updated Q-value (30%).
            adjusted_score = 0.7 * score + 0.3 * new_q
            updated_recommendations.append((movie_id, adjusted_score))
        updated_recommendations.sort(key=lambda x: x[1], reverse=True)
        return updated_recommendations

    def close(self) -> None:
        """
        Close the database connection.
        """
        try:
            self.conn.close()
            logger.info("RLAgent database connection closed.")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")


if __name__ == "__main__":
    # Example usage:
    # Simulated recommendations for user 1.
    agent = RLAgent(learning_rate=0.05, discount_factor=0.95)
    recommendations = [(1, 4.5), (2, 4.2), (3, 3.8)]
    # Simulated feedback: positive feedback for movie 1, negative for movie 2, neutral for movie 3.
    feedback = {1: 1, 2: -1, 3: 0}
    updated_recs = agent.adjust_recommendations(user_id=1, recommendations=recommendations, feedback=feedback)
    print("Updated Recommendations:")
    for movie_id, score in updated_recs:
        print(f"Movie ID: {movie_id}, Adjusted Score: {score:.4f}")
    agent.close()

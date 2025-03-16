import json
import logging
from kafka import KafkaConsumer
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Maximum number of worker threads for concurrent processing
MAX_WORKERS = 5
# Maximum retry attempts for processing a message
MAX_RETRIES = 3

class StreamProcessor:
    def __init__(self, topic: str = "user_behavior", bootstrap_servers: list = None,
                 group_id: str = "stream_processor_group"):
        """
        Initialize the stream processor.

        Args:
            topic (str): Kafka topic to subscribe to.
            bootstrap_servers (list): List of Kafka bootstrap servers.
            group_id (str): Consumer group id.
        """
        if bootstrap_servers is None:
            bootstrap_servers = ["localhost:9092"]
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.consumer = None
        self._initialize_consumer()
        # Thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    def _initialize_consumer(self):
        """
        Initialize the Kafka consumer.
        """
        try:
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                group_id=self.group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8'))
            )
            logger.info(f"Kafka consumer initialized for topic '{self.topic}' on servers {self.bootstrap_servers}.")
        except Exception as e:
            logger.error(f"Error initializing Kafka consumer: {e}")
            self.consumer = None

    def process_message(self, message: dict):
        """
        Process a single message from the Kafka stream with retry mechanism.

        Args:
            message (dict): A dictionary representing the user behavior message.
        """
        attempt = 0
        while attempt < MAX_RETRIES:
            try:
                self.handle_message(message)
                return  # Successful processing; exit the retry loop
            except Exception as e:
                attempt += 1
                logger.exception(f"Error processing message (attempt {attempt}/{MAX_RETRIES}): {e}")
                time.sleep(1)  # brief pause before retrying
        logger.error("Max retry attempts reached. Skipping message.")

    def process_stream(self):
        """
        Continuously process messages from the Kafka topic using a thread pool for concurrency.
        """
        if self.consumer is None:
            logger.error("Consumer not initialized. Exiting stream processing.")
            return

        logger.info("Starting stream processing...")
        futures = []
        try:
            for message in self.consumer:
                msg_value = message.value
                # Submit message processing task to the thread pool
                future = self.executor.submit(self.process_message, msg_value)
                futures.append(future)

                # Optionally, you can check completed futures periodically
                for f in as_completed(futures):
                    try:
                        f.result()  # This will re-raise any exceptions caught in the thread
                    except Exception as e:
                        logger.error(f"Error in a worker thread: {e}")
                    # Remove completed futures from the list
                    futures = [f for f in futures if not f.done()]
        except Exception as e:
            logger.exception(f"Error processing stream: {e}")
        finally:
            self.executor.shutdown(wait=True)
            logger.info("Stream processor shutdown.")

    def handle_message(self, message: dict):
        """
        Process a single message from the Kafka stream.

        Expected message format:
            {
                "user_id": 1,
                "action": "rate" or "click" or "watch",
                "movie_id": 101,
                "rating": 4.5,         # optional, only if action is "rate"
                "timestamp": "2025-02-28T10:00:00Z"
            }
        """
        user_id = message.get("user_id")
        action = message.get("action")
        movie_id = message.get("movie_id")
        timestamp = message.get("timestamp")

        if action == "rate":
            rating = message.get("rating")
            logger.info(f"User {user_id} rated movie {movie_id} with {rating} at {timestamp}.")
            self._update_user_rating(user_id, movie_id, rating, timestamp)
        elif action == "click":
            logger.info(f"User {user_id} clicked on movie {movie_id} at {timestamp}.")
            self._update_user_click(user_id, movie_id, timestamp)
        elif action == "watch":
            logger.info(f"User {user_id} watched movie {movie_id} at {timestamp}.")
            self._update_user_watch(user_id, movie_id, timestamp)
        else:
            logger.warning(f"Unknown action '{action}' received for user {user_id} on movie {movie_id}.")

        # Optionally: invoke additional updates, e.g., updating recommendation model in real-time

    def _update_user_rating(self, user_id: int, movie_id: int, rating: float, timestamp: str):
        """
        Handle updating the system when a user rates a movie.
        """
        # TODO: Integrate with user profile updates or trigger model refresh.
        logger.debug(f"Updating rating for user {user_id}, movie {movie_id}: rating={rating}, timestamp={timestamp}")

    def _update_user_click(self, user_id: int, movie_id: int, timestamp: str):
        """
        Handle updating the system when a user clicks on a movie.
        """
        # TODO: Integrate click tracking or update engagement metrics.
        logger.debug(f"Updating click event for user {user_id}, movie {movie_id} at {timestamp}")

    def _update_user_watch(self, user_id: int, movie_id: int, timestamp: str):
        """
        Handle updating the system when a user watches a movie.
        """
        # TODO: Integrate watch event handling, such as updating viewing history.
        logger.debug(f"Updating watch event for user {user_id}, movie {movie_id} at {timestamp}")

if __name__ == "__main__":
    processor = StreamProcessor(topic="user_behavior", bootstrap_servers=["localhost:9092"])
    processor.process_stream()

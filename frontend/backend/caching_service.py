import logging
import json
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

try:
    import redis

    USE_REDIS = True
    logger.info("Redis module found. Caching will use Redis.")
except ImportError:
    USE_REDIS = False
    logger.warning("Redis module not found. Falling back to in-memory caching.")

# If using Redis, initialize the client (adjust parameters as needed)
if USE_REDIS:
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0)
        redis_client.ping()
        logger.info("Connected to Redis successfully.")
    except Exception as e:
        logger.error(f"Error connecting to Redis: {e}")
        USE_REDIS = False

# In-memory cache fallback
_in_memory_cache = {}


def set_cache(key: str, value: dict, expire: int = 3600) -> None:
    """
    Set a value in the cache with an expiration time.

    Args:
        key (str): The cache key.
        value (dict): The value to store (will be JSON-serialized).
        expire (int): Expiration time in seconds (default: 1 hour).
    """
    try:
        if USE_REDIS:
            redis_client.setex(key, expire, json.dumps(value))
            logger.info(f"Set key '{key}' in Redis cache with expiration {expire}s.")
        else:
            _in_memory_cache[key] = {
                "value": value,
                "expire_at": time.time() + expire
            }
            logger.info(f"Set key '{key}' in in-memory cache with expiration {expire}s.")
    except Exception as e:
        logger.error(f"Error setting cache for key '{key}': {e}")


def get_cache(key: str) -> dict:
    """
    Retrieve a value from the cache.

    Args:
        key (str): The cache key.

    Returns:
        dict: The cached value if found and not expired, otherwise None.
    """
    try:
        if USE_REDIS:
            cached = redis_client.get(key)
            if cached:
                logger.info(f"Key '{key}' found in Redis cache.")
                return json.loads(cached.decode('utf-8'))
            else:
                logger.info(f"Key '{key}' not found in Redis cache.")
                return None
        else:
            entry = _in_memory_cache.get(key)
            if entry and entry["expire_at"] > time.time():
                logger.info(f"Key '{key}' found in in-memory cache.")
                return entry["value"]
            else:
                if key in _in_memory_cache:
                    logger.info(f"Key '{key}' expired in in-memory cache. Removing entry.")
                    del _in_memory_cache[key]
                return None
    except Exception as e:
        logger.error(f"Error retrieving cache for key '{key}': {e}")
        return None


def delete_cache(key: str) -> None:
    """
    Delete a key from the cache.

    Args:
        key (str): The cache key to delete.
    """
    try:
        if USE_REDIS:
            redis_client.delete(key)
            logger.info(f"Key '{key}' deleted from Redis cache.")
        else:
            if key in _in_memory_cache:
                del _in_memory_cache[key]
                logger.info(f"Key '{key}' deleted from in-memory cache.")
    except Exception as e:
        logger.error(f"Error deleting cache for key '{key}': {e}")


if __name__ == "__main__":
    # Example usage:
    test_key = "recommendations_user_1"
    test_value = {"movies": [1, 2, 3, 4, 5], "timestamp": time.time()}

    # Set a cache value
    set_cache(test_key, test_value, expire=60)

    # Retrieve the cache value
    cached_value = get_cache(test_key)
    print(f"Cached Value for '{test_key}':", cached_value)

    # Delete the cache key
    delete_cache(test_key)
    cached_value = get_cache(test_key)
    print(f"Cached Value after deletion for '{test_key}':", cached_value)

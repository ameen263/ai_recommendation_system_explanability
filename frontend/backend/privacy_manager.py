import sqlite3
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrivacyManager:
    def __init__(self, db_path='privacy.db'):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_consent (
                    user_id TEXT PRIMARY KEY,
                    consent BOOLEAN,
                    timestamp DATETIME
                )
            ''')

    def record_consent(self, user_id, consent):
        timestamp = datetime.utcnow()
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO user_consent (user_id, consent, timestamp)
                    VALUES (?, ?, ?)
                ''', (user_id, consent, timestamp))
                conn.commit()
            logging.info(f"Consent recorded successfully for user {user_id}: {consent}")
        except Exception as e:
            logging.error(f"Error recording consent for user {user_id}: {e}")

    def load_consent(self, user_id):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT consent FROM user_consent WHERE user_id=?
                ''', (user_id,))
                result = cursor.fetchone()
                return result[0] if result else None
        except Exception as e:
            logging.exception(f"Error loading consent for user {user_id}: {e}")
            return None

    def get_consent(self, user_id):
        consent = self.load_consent(user_id)
        if consent is None:
            logging.info(f"No consent record found for user {user_id}. Defaulting consent to True for testing.")
            return True
        logging.info(f"Consent retrieved for user {user_id}: {consent}")
        return consent

    def remove_user_data(self, user_id):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM user_consent WHERE user_id=?
                ''', (user_id,))
                conn.commit()
            logging.info(f"Privacy-sensitive data removed successfully for user {user_id}")
        except Exception as e:
            logging.error(f"Error removing user data for user {user_id}: {e}")
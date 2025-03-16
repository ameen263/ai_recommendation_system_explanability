from flask import Flask, jsonify
from api import recommendation_app  # Flask Blueprint
from auth_api import auth_app  # Flask Blueprint
from flask_cors import CORS  # Flask-CORS

# Constants
APP_TITLE = "Unified AI Movie Recommendation API"
APP_VERSION = "1.0"

# Initialize Flask app instance
app = Flask(__name__)
app.config["APP_TITLE"] = APP_TITLE
app.config["APP_VERSION"] = APP_VERSION

# Middleware configuration for CORS
CORS(app, supports_credentials=True)

# Register Flask Blueprints from sub-applications
app.register_blueprint(recommendation_app, url_prefix="/api")
app.register_blueprint(auth_app, url_prefix="/auth")

# Helper configuration flag for startup event
app.config["API_STARTED"] = False


@app.before_request
def startup_event():
    if not app.config["API_STARTED"]:
        print("API has started successfully.")
        app.config["API_STARTED"] = True


# Root endpoint providing basic API guidance
@app.route("/")
def get_root_endpoint():
    return jsonify({
        "message": "Welcome to the Unified API. Use /auth for authentication and /api for recommendations."
    })


# Entry-point
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
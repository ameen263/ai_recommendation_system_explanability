# auth_api.py
from flask import Blueprint, jsonify, request

# Define flask Blueprint clearly and explicitly
auth_app = Blueprint("auth_app", __name__)

# Mocked user data for demonstration purposes (replace this with database interaction)
MOCK_USERS_DB = {
    "user1": {"password": "password123", "user_id": "001", "name": "John Doe"},
    "user2": {"password": "password321", "user_id": "002", "name": "Jane Smith"}
}


# Home endpoint for auth API verification
@auth_app.route("/", methods=["GET"])
def auth_home():
    return jsonify({"message": "Authentication API operational."})


# User login endpoint
@auth_app.route("/login", methods=["POST"])
def login():
    login_data = request.get_json()
    username = login_data.get("username")
    password = login_data.get("password")

    if not username or not password:
        return jsonify({
            "success": False,
            "message": "Username and password are required."
        }), 400

    user = MOCK_USERS_DB.get(username)

    if user and user["password"] == password:
        return jsonify({
            "success": True,
            "message": "Login successful.",
            "user_details": {
                "user_id": user["user_id"],
                "name": user["name"]
            }
        })
    else:
        return jsonify({
            "success": False,
            "message": "Invalid username or password."
        }), 401


# User registration endpoint (simplified example)
@auth_app.route("/register", methods=["POST"])
def register():
    registration_data = request.get_json()
    username = registration_data.get("username")
    password = registration_data.get("password")
    name = registration_data.get("name")

    if not username or not password or not name:
        return jsonify({
            "success": False,
            "message": "Username, password, and name are required."
        }), 400

    if username in MOCK_USERS_DB:
        return jsonify({
            "success": False,
            "message": "Username already exists."
        }), 409

    user_id = str(len(MOCK_USERS_DB) + 1).zfill(3)
    MOCK_USERS_DB[username] = {
        "password": password,
        "user_id": user_id,
        "name": name
    }

    return jsonify({
        "success": True,
        "message": "Registration successful.",
        "user_details": {
            "user_id": user_id,
            "name": name
        }
    }), 201
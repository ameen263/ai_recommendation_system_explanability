# api.py

from flask import Blueprint, jsonify, request
from recommendation_engine import generate_recommendations

# Create a Flask Blueprint for recommendation endpoints
recommendation_app = Blueprint("recommendation_app", __name__)


@recommendation_app.route("/", methods=["GET"])
def home():
    # Simple operational check
    return jsonify({"message": "Recommendation API operational."})


@recommendation_app.route("/advanced_recommend", methods=["POST"])
def advanced_recommend():
    try:
        # Parse JSON request body
        request_data = request.get_json() or {}

        # Required/primary parameters
        user_id = int(request_data.get("user_id", 1))
        num_recommendations = int(request_data.get("num_recommendations", 10))
        include_explainability = bool(request_data.get("include_explainability", True))
        include_fairness_metrics = bool(request_data.get("include_fairness_metrics", True))
        response_format = request_data.get("response_format", "detailed")

        # Optional context and advanced details
        context = request_data.get("context", None)
        explainability_details = request_data.get("explainability_details", {})
        fairness_metrics_request = request_data.get("fairness_metrics", {})
        performance_metrics_request = request_data.get("performance_metrics", {})
        trustworthiness_indicators = request_data.get("trustworthiness_indicators", {})

        # Call the recommendation engine with these parameters
        result = generate_recommendations(
            user_id=user_id,
            context=context,
            include_explainability=include_explainability,
            include_fairness_metrics=include_fairness_metrics,
            response_format=response_format,
            explainability_details=explainability_details,
            fairness_metrics_request=fairness_metrics_request,
            performance_metrics_request=performance_metrics_request,
            trustworthiness_indicators=trustworthiness_indicators
        )

        # Truncate the recommendation list to 'num_recommendations'
        if "recommendations" in result and isinstance(result["recommendations"], list):
            result["recommendations"] = result["recommendations"][:num_recommendations]

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

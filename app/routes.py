from flask import Blueprint, jsonify, request

from app.predict import ChurnPredictor


churn_api = Blueprint("churn_api", __name__)


@churn_api.get("/health")
def health_check():
    try:
        predictor = ChurnPredictor()
        return jsonify(
            {
                "status": "healthy",
                "required_feature_count": len(predictor.required_features),
            }
        )
    except FileNotFoundError as error:
        return jsonify({"status": "unhealthy", "message": str(error)}), 503


@churn_api.post("/predict")
def predict_churn():
    """Return a churn prediction for one JSON customer record."""
    customer_data = request.get_json(silent=True)
    if not isinstance(customer_data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    try:
        result = ChurnPredictor().predict(customer_data)
        return jsonify(result), 200
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 503
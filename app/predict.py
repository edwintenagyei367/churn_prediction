from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("assets/churn_baseline_model.joblib")


class ChurnPredictor:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {model_path}. Run 'python train.py' first."
            )

        self.model = joblib.load(model_path)
        self.required_features = self.model.feature_names_in_.tolist()

    def predict(self, customer_data: dict) -> dict:
        missing_features = sorted(set(self.required_features) - set(customer_data))
        if missing_features:
            raise ValueError(f"Missing required fields: {missing_features}")

        customer_frame = pd.DataFrame(
            [{feature: customer_data[feature] for feature in self.required_features}]
        )

        probability = float(self.model.predict_proba(customer_frame)[0, 1])
        prediction = int(self.model.predict(customer_frame)[0])

        if probability >= 0.70:
            risk_level = "High"
        elif probability >= 0.40:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return {
            "churn_prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(probability, 4),
            "risk_level": risk_level,
        }

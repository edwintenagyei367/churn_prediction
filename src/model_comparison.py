import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluation import EvaluationResults, evaluate_model
from src.training import build_preprocessor


def build_candidate_models(X_train: pd.DataFrame) -> dict[str, Pipeline]:
    """Create candidate models with an independent preprocessor per model."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def compare_models(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    fitted_models: dict[str, Pipeline] = {}
    detailed_results: dict[str, EvaluationResults] = {}
    summary_rows: list[dict[str, float | str]] = []

    for model_name, model in build_candidate_models(X_train).items():
        model.fit(X_train, y_train)
        results = evaluate_model(model, X_test, y_test)

        fitted_models[model_name] = model
        detailed_results[model_name] = results
        summary_rows.append(
            {
                "Model": model_name,
                "Accuracy": results.accuracy,
                "Precision": results.precision,
                "Recall": results.recall,
                "F1-score": results.f1_score,
                "ROC-AUC": results.roc_auc,
            }
        )

    comparison_table = pd.DataFrame(summary_rows).sort_values(
        by=["ROC-AUC", "F1-score"], ascending=False
    )
    return fitted_models, detailed_results, comparison_table

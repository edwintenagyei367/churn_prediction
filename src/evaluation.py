from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvaluationResults:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: list[list[int]]
    classification_report: dict

    def save(self, path: Path) -> None:
        """Save model results as a JSON artifact."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    predictions = model.predict(X_test)

    if not hasattr(model, "predict_proba"):
        raise TypeError("The model must support predict_proba to calculate ROC-AUC.")
    probabilities = model.predict_proba(X_test)[:, 1]

    return EvaluationResults(
        accuracy=float(accuracy_score(y_test, predictions)),
        precision=float(precision_score(y_test, predictions, zero_division=0)),
        recall=float(recall_score(y_test, predictions, zero_division=0)),
        f1_score=float(f1_score(y_test, predictions, zero_division=0)),
        roc_auc=float(roc_auc_score(y_test, probabilities)),
        confusion_matrix=confusion_matrix(y_test, predictions).astype(int).tolist(),
        classification_report=classification_report(
            y_test,
            predictions,
            target_names=["Retained", "Churned"],
            output_dict=True,
            zero_division=0,
        ),
    )


def print_evaluation(results: EvaluationResults) -> None:
    print("Model Evaluation")
    print(f"Accuracy:  {results.accuracy:.3f}")
    print(f"Precision: {results.precision:.3f}")
    print(f"Recall:    {results.recall:.3f}")
    print(f"F1-score:  {results.f1_score:.3f}")
    print(f"ROC-AUC:   {results.roc_auc:.3f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(np.array(results.confusion_matrix))

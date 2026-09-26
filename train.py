import json
from pathlib import Path

import joblib

from src.data_ingestion import DataIngestion, DataIngestionConfig
from src.data_validation import DataValidator
from src.evaluation import evaluate_model, print_evaluation
from src.training import build_baseline_pipeline, split_data,TrainingConfig
from src.preprocessing import prepare_data


ASSETS_DIR = Path("assets")
MODEL_PATH = ASSETS_DIR / "churn_baseline_model.joblib"
METRICS_PATH = ASSETS_DIR / "baseline_metrics.json"
METADATA_PATH = ASSETS_DIR / "model_metadata.json"
VALIDATION_REPORT_PATH = ASSETS_DIR / "validation_report.json"


def main():
    # 1. Ingest raw data.
    raw_data = DataIngestion(DataIngestionConfig()).run()

    # 2. Validate data before training.
    validation_report = DataValidator().validate(raw_data)
    validation_report.save(VALIDATION_REPORT_PATH)
    if not validation_report.passed:
        raise ValueError(
            "Data validation failed. Review artifacts/validation_report.json before training."
        )

    # 3. Create leakage-safe features and a binary churn target.
    prepared_data = prepare_data(raw_data)

    # 4. Split data before fitting encoders and scalers.
    X_train, X_test, y_train, y_test = split_data(prepared_data,TrainingConfig)

    # 5. Fit the full preprocessing + Logistic Regression pipeline.
    model = build_baseline_pipeline(X_train)
    model.fit(X_train, y_train)

    # 6. Evaluate only on the unseen test set.
    results = evaluate_model(model, X_test, y_test)
    print_evaluation(results)

    # 7. Save the fitted pipeline and its performance record.
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    results.save(METRICS_PATH)

    metadata = {
        "model_name": "LogisticRegression",
        "model_file": str(MODEL_PATH),
        "target": "Churn Label (No=0, Yes=1)",
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count_before_encoding": int(X_train.shape[1]),
        "random_state": 42,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()

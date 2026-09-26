from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.preprocessing import CleanedData

@dataclass(frozen=True)
class TrainingConfig:
    test_size = 0.20
    random_state=42

def split_data(clean_data, config):
    config = TrainingConfig()
    return train_test_split(
        clean_data.X,
        clean_data.y,
        test_size=config.test_size,
        random_state= config.random_state,
        stratify=clean_data.y
    )

def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build transformations using feature types observed in training data only."""
    numeric_features = X_train.select_dtypes(include="number").columns.tolist()
    categorical_features = X_train.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def build_baseline_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """Create an end-to-end Logistic Regression churn-classification pipeline."""
    preprocessor = build_preprocessor(X_train)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

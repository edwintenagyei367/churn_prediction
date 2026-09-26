from dataclasses import dataclass
import pandas as pd

TARGET_COLUMN = "Churn Label"

DROP_COLUMNS = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Value",
    "Churn Score",
    "CLTV",
    "Churn Reason",
]

@dataclass
class CleanedData:
    X : pd.DataFrame
    y : pd.Series

def prepare_data(data: pd.DataFrame):
    clean_data = data.copy()
    clean_data.columns = clean_data.columns.str.strip()

    if TARGET_COLUMN not in clean_data.columns:
        raise ValueError(f"Required target column'{TARGET_COLUMN}' was not found.")

    if "Total Charges" in clean_data.columns:
        clean_data["Total Charges"] = pd.to_numeric(clean_data["Total Charges"], errors="coerce")
        clean_data = clean_data.dropna(subset=["Total Charges"])

    y = clean_data[TARGET_COLUMN].map({"No":0, "Yes": 1})
    if y.isna().any():
        unexpected = clean_data.loc[y.isna(), TARGET_COLUMN]. unique().tolist()
        raise ValueError(
            f"Unexpected values: {unexpected}"
        )

    X = clean_data.drop(columns = DROP_COLUMNS + [TARGET_COLUMN], errors="ignore")
    if X.empty:
        raise ValueError("No model features remain afater preprocessing.")

    return CleanedData(X=X, y=y.astype(int))


from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import json

import pandas as pd


REQUIRED_COLUMNS = {
    "CustomerID",
    "Gender",
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "Churn Label",
}


@dataclass
class ValidationReport:
    passed: bool
    row_count: int
    column_count: int
    duplicate_rows: int
    missing_required_columns: list[str]
    invalid_churn_labels: list[str]
    missing_values: dict[str, int]
    errors: list[str]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


class DataValidator:
    def __init__(
        self,
        required_columns: Iterable[str] = REQUIRED_COLUMNS,
        target_column: str = "Churn Label",
    ):
        self.required_columns = set(required_columns)
        self.target_column = target_column

    def validate(self, data: pd.DataFrame):
        """Validate structure and quality without changing the source data."""
        errors: list[str] = []
        missing_columns = sorted(self.required_columns - set(data.columns))
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")

        invalid_labels: list[str] = []
        if self.target_column in data.columns:
            labels = set(data[self.target_column].dropna().astype(str).str.strip().unique())
            invalid_labels = sorted(labels - {"Yes", "No"})
            if invalid_labels:
                errors.append(
                    f"{self.target_column} must contain only 'Yes' or 'No'; "
                    f"found {invalid_labels}."
                )

        for numeric_column in ("Tenure Months", "Monthly Charges"):
            if numeric_column in data.columns:
                converted = pd.to_numeric(data[numeric_column], errors="coerce")
                if converted.isna().any():
                    errors.append(f"{numeric_column} contains missing or non-numeric values.")
                elif (converted < 0).any():
                    errors.append(f"{numeric_column} contains negative values.")

        if "Total Charges" in data.columns:
            # Total Charges can contain blank strings in the raw Telco dataset.
            converted = pd.to_numeric(data["Total Charges"], errors="coerce")
            if (converted.dropna() < 0).any():
                errors.append("Total Charges contains negative values.")

        missing_values = {
            column: int(count)
            for column, count in data.isna().sum().items()
            if count > 0
        }

        return ValidationReport(
            passed=not errors,
            row_count=len(data),
            column_count=len(data.columns),
            duplicate_rows=int(data.duplicated().sum()),
            missing_required_columns=missing_columns,
            invalid_churn_labels=invalid_labels,
            missing_values=missing_values,
            errors=errors,
        )

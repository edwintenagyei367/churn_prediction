from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DataIngestionConfig:
    raw_data_path: Path = Path("data/raw/churn.csv")
    ingested_data_path: Path = Path("data/processed/cleaned_churn.csv")


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def run(self) -> pd.DataFrame:
        if not self.config.raw_data_path.exists():
            raise FileNotFoundError(
                f"Raw data file not found: {self.config.raw_data_path}. "
                "Place the original Telco CSV in data/raw/."
            )

        data = pd.read_csv(self.config.raw_data_path)
        data.columns = data.columns.str.strip()

        self.config.ingested_data_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(self.config.ingested_data_path, index=False)
        return data

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    raw_data_path: Path = ROOT / "data" / "raw" / "california_housing_sample.csv"
    processed_data_path: Path = ROOT / "data" / "processed" / "featured_housing.csv"
    model_path: Path = ROOT / "artifacts" / "house_price_model.joblib"
    metadata_path: Path = ROOT / "artifacts" / "model_metadata.json"


SETTINGS = Settings()

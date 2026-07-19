"""Housing data validation and deterministic California-style fallback data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    "longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
    "population", "households", "median_income",
]
REQUIRED_COLUMNS = set(NUMERIC_COLUMNS + ["ocean_proximity", "median_house_value"])


def validate_housing(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    clean = frame.copy()
    for column in NUMERIC_COLUMNS + ["median_house_value"]:
        clean[column] = pd.to_numeric(clean[column], errors="raise")
    if (clean[["total_rooms", "population", "households", "median_income", "median_house_value"]] <= 0).any().any():
        raise ValueError("Room, population, household, income, and target values must be positive")
    if not clean["ocean_proximity"].dropna().map(lambda value: isinstance(value, str)).all():
        raise ValueError("ocean_proximity must be text when supplied")
    return clean.reset_index(drop=True)


def generate_sample_housing(seed: int = 42, rows: int = 900) -> pd.DataFrame:
    """Generate a deterministic California-housing-shaped offline sample."""
    rng = np.random.default_rng(seed)
    income = rng.lognormal(1.25, 0.45, rows).clip(0.7, 15)
    latitude = rng.uniform(32.5, 41.8, rows)
    longitude = rng.uniform(-124.2, -114.0, rows)
    age = rng.integers(1, 53, rows)
    households = rng.integers(120, 1_800, rows)
    rooms_per_household = rng.uniform(3.5, 8.5, rows)
    total_rooms = (households * rooms_per_household).round().astype(int)
    bedrooms = (total_rooms * rng.uniform(0.15, 0.32, rows)).round().astype(float)
    population = (households * rng.uniform(1.5, 4.4, rows)).round().astype(int)
    proximity = rng.choice(["INLAND", "NEAR BAY", "NEAR OCEAN", "<1H OCEAN", "ISLAND"], rows, p=[.38, .16, .18, .26, .02])
    coastal_premium = np.select([proximity == "ISLAND", proximity == "NEAR BAY", proximity == "NEAR OCEAN", proximity == "<1H OCEAN"], [115_000, 72_000, 50_000, 30_000], default=0)
    value = 42_000 + income * 42_000 + coastal_premium + (42 - age) * 650 + rng.normal(0, 23_000, rows)
    frame = pd.DataFrame({"longitude": longitude, "latitude": latitude, "housing_median_age": age, "total_rooms": total_rooms, "total_bedrooms": bedrooms, "population": population, "households": households, "median_income": income, "ocean_proximity": proximity, "median_house_value": value.clip(25_000, 500_000)})
    # Exercise imputation without affecting target or validation semantics.
    frame.loc[rng.choice(rows, size=max(1, rows // 25), replace=False), "total_bedrooms"] = np.nan
    frame.loc[rng.choice(rows, size=max(1, rows // 40), replace=False), "ocean_proximity"] = np.nan
    return validate_housing(frame)


def load_housing(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Housing data not found: {source}")
    return validate_housing(pd.read_csv(source))

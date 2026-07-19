"""Deterministic feature engineering and reproducible train/test partitioning."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from house_price_predictor.data import NUMERIC_COLUMNS


TARGET = "median_house_value"
CATEGORICAL_COLUMNS = ["ocean_proximity"]
ENGINEERED_COLUMNS = ["rooms_per_household", "bedrooms_per_room", "population_per_household"]


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["rooms_per_household"] = data["total_rooms"] / data["households"]
    data["bedrooms_per_room"] = data["total_bedrooms"] / data["total_rooms"]
    data["population_per_household"] = data["population"] / data["households"]
    return data


def split_data(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(frame, test_size=0.2, random_state=seed)
    return train.sort_index().copy(), test.sort_index().copy()


def model_columns() -> tuple[list[str], list[str]]:
    return NUMERIC_COLUMNS + ENGINEERED_COLUMNS, CATEGORICAL_COLUMNS

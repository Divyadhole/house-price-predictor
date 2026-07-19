"""Fitted preprocessing and regression models stored as one safe inference artifact."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from house_price_predictor.features import TARGET, model_columns


def _preprocessor() -> ColumnTransformer:
    numeric, categorical = model_columns()
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("category", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])


@dataclass
class HousePriceModel:
    baseline: Pipeline
    tuned: Pipeline
    selected: str = "tuned"

    @classmethod
    def fit(cls, train: pd.DataFrame) -> "HousePriceModel":
        x, y = train.drop(columns=[TARGET]), train[TARGET]
        baseline = Pipeline([("preprocess", _preprocessor()), ("model", Ridge(alpha=5.0))]).fit(x, y)
        tuned = Pipeline([("preprocess", _preprocessor()), ("model", HistGradientBoostingRegressor(max_iter=220, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, random_state=42))]).fit(x, y)
        return cls(baseline=baseline, tuned=tuned)

    def predict(self, frame: pd.DataFrame, model: str = "selected") -> pd.Series:
        chosen = self.selected if model == "selected" else model
        pipeline = self.tuned if chosen == "tuned" else self.baseline
        prediction = pipeline.predict(frame.drop(columns=[TARGET], errors="ignore"))
        return pd.Series(prediction, index=frame.index, name="predicted_value")

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: str | Path) -> "HousePriceModel":
        model = joblib.load(Path(path))
        if not isinstance(model, cls):
            raise TypeError("Artifact is not a HousePriceModel")
        return model

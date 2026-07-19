"""Regression metrics and interpretable residual summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    if len(actual) != len(predicted) or not np.isfinite(predicted).all():
        raise ValueError("Actuals and finite predictions must have matching lengths")
    return {
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def residual_frame(actual: pd.Series, predicted: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"actual": actual, "predicted": predicted, "residual": actual - predicted})

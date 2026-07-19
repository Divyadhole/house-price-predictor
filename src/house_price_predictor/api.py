"""Typed FastAPI inference application for house value estimates."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from house_price_predictor.config import SETTINGS
from house_price_predictor.features import add_features
from house_price_predictor.model import HousePriceModel


class HouseRequest(BaseModel):
    longitude: float = Field(ge=-125, le=-113)
    latitude: float = Field(ge=32, le=43)
    housing_median_age: float = Field(ge=1, le=100)
    total_rooms: float = Field(gt=0)
    total_bedrooms: float | None = Field(default=None, gt=0)
    population: float = Field(gt=0)
    households: float = Field(gt=0)
    median_income: float = Field(gt=0, le=30)
    ocean_proximity: str | None = None


@lru_cache(maxsize=1)
def load_default_model() -> HousePriceModel:
    if not SETTINGS.model_path.exists():
        raise RuntimeError("Model artifact is missing. Run `house-price train` first.")
    return HousePriceModel.load(SETTINGS.model_path)


def create_app(model: HousePriceModel | None = None) -> FastAPI:
    app = FastAPI(title="House Price Predictor", version="1.0.0")
    app.state.model = model

    def current_model() -> HousePriceModel:
        return app.state.model or load_default_model()

    @app.get("/health")
    def health() -> dict[str, str]:
        try:
            current_model()
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return {"status": "healthy", "model": "housing-regression-v1"}

    @app.post("/predict")
    def predict(request: HouseRequest) -> dict[str, float]:
        frame = add_features(pd.DataFrame([request.model_dump()]))
        return {"estimated_house_value": round(float(current_model().predict(frame).iloc[0]), 2)}

    return app


app = create_app()

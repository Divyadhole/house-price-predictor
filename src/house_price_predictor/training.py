"""End-to-end reproducible training, comparison, and artifact persistence."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from house_price_predictor.config import SETTINGS, Settings
from house_price_predictor.data import load_housing
from house_price_predictor.evaluation import regression_metrics, residual_frame
from house_price_predictor.features import TARGET, add_features, split_data
from house_price_predictor.model import HousePriceModel


def train_and_evaluate(settings: Settings = SETTINGS) -> dict[str, object]:
    featured = add_features(load_housing(settings.raw_data_path))
    train, test = split_data(featured, settings.seed)
    model = HousePriceModel.fit(train)
    baseline_prediction = model.predict(test, "baseline")
    tuned_prediction = model.predict(test, "tuned")
    baseline_metrics = regression_metrics(test[TARGET], baseline_prediction)
    tuned_metrics = regression_metrics(test[TARGET], tuned_prediction)
    chosen = "tuned" if tuned_metrics["rmse"] <= baseline_metrics["rmse"] else "baseline"
    chosen_prediction = tuned_prediction if chosen == "tuned" else baseline_prediction
    settings.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(settings.processed_data_path, index=False)
    residual_frame(test[TARGET], chosen_prediction).to_csv(settings.processed_data_path.parent / "residuals.csv", index=False)
    model.save(settings.model_path)
    metadata: dict[str, object] = {
        "trained_at": datetime.now(UTC).isoformat(), "seed": settings.seed,
        "rows": len(featured), "train_rows": len(train), "test_rows": len(test),
        "baseline_metrics": baseline_metrics, "tuned_metrics": tuned_metrics,
        "selected_model": chosen,
    }
    settings.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    settings.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata

import json

from fastapi.testclient import TestClient

from house_price_predictor.api import create_app
from house_price_predictor.cli import main
from house_price_predictor.data import generate_sample_housing
from house_price_predictor.features import add_features, split_data
from house_price_predictor.model import HousePriceModel


def _model_and_request():
    data = add_features(generate_sample_housing(rows=300))
    train, test = split_data(data, 42)
    model = HousePriceModel.fit(train)
    row = test.iloc[0]
    request = {key: (None if row[key] != row[key] else row[key]) for key in ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity"]}
    return model, request


def test_api_health_prediction_and_bad_input():
    model, request = _model_and_request()
    client = TestClient(create_app(model))
    assert client.get("/health").status_code == 200
    assert client.post("/predict", json=request).status_code == 200
    assert client.post("/predict", json={**request, "median_income": -1}).status_code == 422


def test_cli_smoke_prediction(tmp_path, capsys):
    model, request = _model_and_request()
    artifact, payload = tmp_path / "model.joblib", tmp_path / "request.json"
    model.save(artifact)
    payload.write_text(json.dumps(request))
    assert main(["predict", str(payload), "--model", str(artifact)]) == 0
    assert "estimated_house_value" in capsys.readouterr().out

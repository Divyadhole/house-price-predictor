import numpy as np

from house_price_predictor.data import generate_sample_housing
from house_price_predictor.features import add_features, split_data
from house_price_predictor.model import HousePriceModel


def _train_model():
    featured = add_features(generate_sample_housing(rows=300))
    train, test = split_data(featured, 42)
    return HousePriceModel.fit(train), train, test


def test_split_is_reproducible_and_preprocessing_is_train_fitted():
    data = add_features(generate_sample_housing(rows=300))
    train_a, test_a = split_data(data, 42)
    train_b, test_b = split_data(data, 42)
    assert train_a.index.equals(train_b.index)
    assert not set(train_a.index).intersection(test_a.index)
    model = HousePriceModel.fit(train_a)
    median = model.baseline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"].statistics_[4]
    assert median == train_a["total_bedrooms"].median()
    assert test_a.index.equals(test_b.index)


def test_reload_prediction_matches_original(tmp_path):
    model, _, test = _train_model()
    before = model.predict(test.head(3)).to_numpy()
    path = tmp_path / "model.joblib"
    model.save(path)
    after = HousePriceModel.load(path).predict(test.head(3)).to_numpy()
    assert np.allclose(before, after)

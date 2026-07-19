import pytest

from house_price_predictor.data import generate_sample_housing, validate_housing


def test_synthetic_sample_is_deterministic_and_contains_missing_values():
    first = generate_sample_housing(seed=7, rows=100)
    second = generate_sample_housing(seed=7, rows=100)
    assert first.equals(second)
    assert first["total_bedrooms"].isna().any()


def test_invalid_positive_constraints_are_rejected():
    frame = generate_sample_housing(rows=20)
    frame.loc[0, "median_income"] = 0
    with pytest.raises(ValueError, match="positive"):
        validate_housing(frame)

from house_price_predictor.config import SETTINGS
from house_price_predictor.data import generate_sample_housing

if __name__ == "__main__":
    data = generate_sample_housing(SETTINGS.seed)
    SETTINGS.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(SETTINGS.raw_data_path, index=False)
    print(f"Wrote {len(data)} rows to {SETTINGS.raw_data_path}")

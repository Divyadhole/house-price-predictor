import json

from house_price_predictor.training import train_and_evaluate

if __name__ == "__main__":
    print(json.dumps(train_and_evaluate(), indent=2))

"""Train and predict from the shell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from house_price_predictor.config import SETTINGS
from house_price_predictor.features import add_features
from house_price_predictor.model import HousePriceModel
from house_price_predictor.training import train_and_evaluate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="house-price")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train")
    predict = commands.add_parser("predict")
    predict.add_argument("request", type=Path)
    predict.add_argument("--model", type=Path, default=SETTINGS.model_path)
    args = parser.parse_args(argv)
    if args.command == "train":
        print(json.dumps(train_and_evaluate(), indent=2))
        return 0
    request = json.loads(args.request.read_text())
    estimate = HousePriceModel.load(args.model).predict(add_features(pd.DataFrame([request]))).iloc[0]
    print(json.dumps({"estimated_house_value": round(float(estimate), 2)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

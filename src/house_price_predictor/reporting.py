"""Generate compact EDA and residual diagnostics from committed pipeline outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from house_price_predictor.config import ROOT, SETTINGS


def generate_reports(output_dir: Path = ROOT / "reports" / "figures") -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SETTINGS.processed_data_path)
    residuals = pd.read_csv(SETTINGS.processed_data_path.parent / "residuals.csv")
    output = [_value_distribution(data, output_dir / "value-distribution.png"), _residual_plot(residuals, output_dir / "residuals.png")]
    (ROOT / "reports" / "metrics-summary.json").write_text(SETTINGS.metadata_path.read_text())
    return output


def _value_distribution(data: pd.DataFrame, destination: Path) -> Path:
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.hist(data["median_house_value"], bins=30, color="#12624f", edgecolor="white")
    axis.set(title="Sample house-value distribution", xlabel="Median house value", ylabel="Rows")
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination


def _residual_plot(residuals: pd.DataFrame, destination: Path) -> Path:
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.scatter(residuals["actual"], residuals["residual"], alpha=.65, color="#58758a")
    axis.axhline(0, color="#d6a84b", linestyle="--")
    axis.set(title="Selected-model residuals", xlabel="Actual house value", ylabel="Residual")
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination

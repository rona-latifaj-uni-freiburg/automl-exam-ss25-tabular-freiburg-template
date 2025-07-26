#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path

import pandas as pd

from data import get_test_data
from bootstrap_ensemble import EnsemblePFN


def run_prediction(
    dataset_dir: str,
    model_file: str,
    output_file: str = "y_pred.csv",
    fold: int | None = None,
):
    dataset_dir = Path(dataset_dir)
    model_path = Path(model_file)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load test features for the specified fold (or last-simulated if None)
    X_test = get_test_data(str(dataset_dir), fold=fold)

    # Load the trained ensemble model
    with open(model_path, "rb") as f:
        ensemble: EnsemblePFN = pickle.load(f)

    # Predict
    y_pred = ensemble.predict(X_test)

    # Save predictions
    pred_df = pd.DataFrame({"y_pred": y_pred})
    pred_df.to_csv(output_file, index=False)
    print(f"Saved predictions to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run prediction using a specific TabPFN ensemble pickle"
    )
    parser.add_argument(
        "-d", "--dataset", required=True,
        help="Path to dataset root (contains fold subfolders)"
    )
    parser.add_argument(
        "-m", "--model-file", required=True,
        help="Path to the ensemble.pkl file"
    )
    parser.add_argument(
        "-f", "--fold", type=int, default=None,
        help="Fold number to use (overrides any previously simulated fold)"
    )
    parser.add_argument(
        "-o", "--output", default="y_pred.csv",
        help="CSV file to save predictions"
    )

    args = parser.parse_args()
    run_prediction(
        dataset_dir=args.dataset,
        model_file=args.model_file,
        output_file=args.output,
        fold=args.fold,
    )

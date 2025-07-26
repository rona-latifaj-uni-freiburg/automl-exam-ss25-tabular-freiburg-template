#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path

import pandas as pd

from data import simulate_exam_dataset, get_test_data
from bootstrap_ensemble import EnsemblePFN


def run_prediction(
    dataset_path: str,
    model_dir: str,
    output_file: str = "y_pred.csv"
):
    dataset_path = Path(dataset_path)
    model_dir = Path(model_dir)
    model_path = model_dir / dataset_path.name / "ensemble.pkl"

    # Simulate one fold to set the global _simulated_fold
    simulate_exam_dataset(str(dataset_path))

    # Get the X_test corresponding to the fold
    X_test = get_test_data(str(dataset_path))

    # Load the trained ensemble model
    with open(model_path, "rb") as f:
        ensemble: EnsemblePFN = pickle.load(f)

    # Predict
    y_pred = ensemble.predict(X_test)

    # Save predictions
    pred_df = pd.DataFrame(y_pred, columns=["y_pred"])
    pred_df.to_csv(output_file, index=False)
    print(f"Saved predictions to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction using trained TabPFN ensemble")
    parser.add_argument("-d", "--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("-m", "--model-dir", required=True, help="Directory where ensemble.pkl is saved")
    parser.add_argument("-o", "--output", default="y_pred.csv", help="CSV file to save predictions")

    args = parser.parse_args()
    run_prediction(args.dataset, args.model_dir, args.output)

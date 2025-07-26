#!/usr/bin/env python3

import argparse
import pandas as pd
from sklearn.metrics import r2_score


def load_column(file_path: str, column_name: str = None) -> pd.Series:
    """
    Loads a single-column Series from a CSV or Parquet file.
    If column_name is provided, use that column.
    If not, infer the first column.
    """
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        raise ValueError("Unsupported file format: must be .csv or .parquet")

    if column_name and column_name in df.columns:
        return df[column_name]
    return df.iloc[:, 0]  # fallback to first column


def main(pred_path: str, true_path: str):
    y_pred = load_column(pred_path, column_name="y_pred")
    y_true = load_column(true_path)

    if len(y_pred) != len(y_true):
        raise ValueError(f"Mismatch in number of rows: y_pred={len(y_pred)}, y_true={len(y_true)}")

    r2 = r2_score(y_true, y_pred)
    print(f"✅ R² Score: {r2:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate R² score between predictions and ground truth")
    parser.add_argument("--pred", required=True, help="Path to y_pred.csv")
    parser.add_argument("--true", required=True, help="Path to y_test.parquet or y_test.csv")

    args = parser.parse_args()
    main(args.pred, args.true)

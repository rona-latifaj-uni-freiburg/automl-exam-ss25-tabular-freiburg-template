# src/data/preprocess.py

from pathlib import Path
import numpy as np
import pandas as pd
from src.data.load import (
    get_available_datasets,
    get_available_folds,
    load_fold,
    load_all_folds,
    load_full_train_for_specific_dataset,
    load_full_test_for_specific_dataset,
)


def print_dataset_info():
    """
    Demonstrate and print outputs of all load.py methods.
    """
    datasets = get_available_datasets()
    print("Available datasets:", datasets)

    for ds in datasets:
        folds = get_available_folds(ds)
        print(f"\nDataset '{ds}' has folds:", folds)

        # 1) Single fold
        print(f"\n— Fold {folds[0]} of '{ds}' —")
        X_tr, X_te, y_tr, y_te = load_fold(ds, folds[0])
        print("X_train:\n", X_tr)
        print("y_train:\n", y_tr)
        print("X_test:\n", X_te)
        print("y_test:\n", y_te)

        # 2) All folds dict
        all_folds = load_all_folds(ds)
        print(f"\nLoaded all folds for '{ds}', keys:", list(all_folds.keys()))

        # 3) Full train set
        X_full_tr, y_full_tr = load_full_train_for_specific_dataset(ds)
        print(f"\nFull TRAIN for '{ds}':\n", X_full_tr)
        print("Full y_train:\n", y_full_tr)

        # 4) Full test set
        X_full_te, y_full_te = load_full_test_for_specific_dataset(ds)
        print(f"\nFull TEST for '{ds}':\n", X_full_te)
        print("Full y_test:\n", y_full_te)

        print("\n" + "="*80 + "\n")

def preprocess_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    General-purpose feature preprocessing:
    - One-hot encodes categorical variables
    - Applies log1p to positive numeric columns
    - Fills missing values
    - Aligns train and test feature columns
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Identify numeric and categorical columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # --- Log-transform positive numeric features ---
    def safe_log1p(df, cols):
        for col in cols:
            if (df[col] > 0).all():
                df[col] = np.log1p(df[col])
        return df

    X_train = safe_log1p(X_train, numeric_cols)
    X_test = safe_log1p(X_test, numeric_cols)

    # --- One-hot encode categoricals ---
    X_train = pd.get_dummies(X_train, columns=categorical_cols, dummy_na=True)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, dummy_na=True)

    # --- Align train and test columns ---
    X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=0)

    # --- Fill any remaining NaNs ---
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    return X_train, X_test

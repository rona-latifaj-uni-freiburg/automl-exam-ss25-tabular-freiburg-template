# src/data/preprocess.py

from pathlib import Path
from .load import (
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


if __name__ == "__main__":
    print_dataset_info()

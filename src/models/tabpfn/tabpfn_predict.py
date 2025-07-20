#!/usr/bin/env python3
"""
Predict on a held‑out test set with either
- a single TabPFN fold model (.pkl) or
- an EnsemblePFN saved as ensemble.pkl (or a directory that contains it).

For practice datasets (which *do* ship with y_test) the script also
computes R² and stores it in <output>.r2.txt.

Examples
--------
Single fold:
    python predict.py -d data/bike_sharing_demand \
        -m models/fold5.pkl -o preds/single.csv

Whole ensemble (file or directory):
    python predict.py -d data/bike_sharing_demand \
        -m models/ensemble.pkl -o preds/ensemble.csv
"""
import argparse
import gzip
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import r2_score

from data import load_full_test_for_specific_dataset
from tabpfn_ensemble import EnsemblePFN         # ← adjust import path if needed


# -----------------------------------------------------------------------------#
# Helper functions                                                             #
# -----------------------------------------------------------------------------#
def _hydrate_ensemble(ens: "EnsemblePFN", search_dir: Path):
    """Populate an EnsemblePFN with fold models found on disk (if empty)."""
    if ens.models:                              # already populated
        return ens

    fold_files = sorted(search_dir.glob("*_fold*.pkl"))
    if not fold_files:
        raise ValueError(
            "EnsemblePFN contains no fold models and none were found on disk.\n"
            "Re‑export the ensemble *after* training, or provide the fold files."
        )
    ens.models = [pickle.load(fp.open("rb")) for fp in fold_files]
    return ens


def _load_model(path: Path):
    """
    path:
        • directory → tries <dir>/ensemble.pkl, else builds ensemble from fold files
        • *.pkl/.pkl.gz → loads single fold or ensemble directly
    """
    if path.is_dir():
        ens_pkl = path / "ensemble.pkl"
        if ens_pkl.exists():
            with ens_pkl.open("rb") as f:
                obj = pickle.load(f)
            return _hydrate_ensemble(obj, path) if isinstance(obj, EnsemblePFN) else obj

        # no ensemble.pkl → try fold files directly
        folds = sorted(path.glob("*_fold*.pkl"))
        if not folds:
            raise FileNotFoundError(f"No ensemble.pkl or *_fold*.pkl files found in {path}")
        return EnsemblePFN([pickle.load(fp.open("rb")) for fp in folds])

    # path is a file ----------------------------------------------------------
    if path.suffix not in {".pkl", ".pkl.gz"}:
        raise ValueError("model‑path must be a .pkl/.pkl.gz file or a directory")

    with (path.open("rb") if path.suffix == ".pkl" else gzip.open(path, "rb")) as f:
        obj = pickle.load(f)
    return _hydrate_ensemble(obj, path.parent) if isinstance(obj, EnsemblePFN) else obj
# -----------------------------------------------------------------------------#


def main(dataset_dir: str, model_path: Path, output_path: Path) -> None:
    """Load data, predict, save CSV, and (if possible) compute R²."""
    # -------- data -----------------------------------------------------------
    X_test, y_test = load_full_test_for_specific_dataset(dataset_dir)
    y_test = y_test.squeeze("columns")          # Series (N,), not DataFrame

    # -------- model ----------------------------------------------------------
    model = _load_model(model_path)
    y_pred = model.predict(X_test).squeeze()    # ndarray (N,)

    # -------- save predictions ----------------------------------------------
    out = pd.DataFrame({
        "prediction": y_pred,
        "target":     y_test.reset_index(drop=True)
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"[+] Predictions saved to {output_path}")

    # -------- optional R² ----------------------------------------------------
    try:
        score = r2_score(y_test, y_pred)
    except Exception as exc:                    # e.g. y_test missing / wrong len
        print(f"[!] Could not compute R²: {exc}")
        return

    r2_path = output_path.with_suffix(".r2.txt")
    r2_path.write_text(f"{score:.6f}\n")
    print(f"[+] R² = {score:.6f}  (saved to {r2_path})")


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("-d", "--dataset", required=True,
                     help="Dataset folder used for training (under data/)")
    cli.add_argument("-m", "--model-path", required=True, type=Path,
                     help="Path to fold/ensemble .pkl or a directory")
    cli.add_argument("-o", "--output-path", required=True, type=Path,
                     help="CSV file to write predictions")
    args = cli.parse_args()
    main(args.dataset, args.model_path, args.output_path)

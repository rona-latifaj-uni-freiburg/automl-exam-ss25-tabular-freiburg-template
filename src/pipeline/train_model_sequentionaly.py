#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import joblib

from models.bootstrap_tabpfn.bootstrap_tabpfn_train import train_bootstrap
from models.tree_based_methods.auto_ml_pipeline_project.final_train import tree_based_methods_model
from models.tabnet.tabnet_pipeline import tabnet_model


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class WeightedEnsemble:
    """
    Wraps models supporting .predict(X) to return a weighted average of predictions.
    """
    def __init__(self, models, weights):
        self.models = models
        w = np.array(weights, dtype=float)
        self.weights = w / w.sum()

    def predict(self, X):
        preds = [m.predict(X) for m in self.models]
        return np.tensordot(self.weights, preds, axes=[0, 0])


def load_model(path: Path):
    """
    Load a model by file extension (.pkl/.joblib via joblib, .pt/.pth via torch).
    """
    ext = path.suffix.lower()
    if ext in {'.pkl', '.joblib'}:
        return joblib.load(path)
    if ext in {'.pt', '.pth'}:
        return torch.load(path, map_location='cpu')
    return joblib.load(path)


def train_and_ensemble(dataset: Path, output_dir: Path, seed: int = 1):
    # 1) Tree-based model
    print("[INFO] Training tree-based model...")
    tree_path, tree_r2 = tree_based_methods_model(
        str(dataset), str(output_dir)
    )
    print(f"Tree-based → {tree_path} (R²={tree_r2:.4f})")

    # 3) TabNet model
    print("[INFO] Training TabNet model...")
    tabnet_dir = output_dir / "tabnet"
    tabnet_dir.mkdir(parents=True, exist_ok=True)
    tabnet_path, tabnet_r2 = tabnet_model(
        str(dataset), str(tabnet_dir),
        n_splits=10, n_trials=3, seed=seed
    )
    print(f"TabNet →      {tabnet_path} (mean R²={tabnet_r2:.4f})")
    
    # 2) TabPFN model
    print("[INFO] Training TabPFN model...")
    tabpfn_path, tabpfn_r2 = train_bootstrap(
        str(dataset), output_dir=output_dir, seed=seed,
        use_optuna=True, n_trials=50, fold=1
    )
    print(f"TabPFN →      {tabpfn_path} (R²={tabpfn_r2:.4f})")


    # 4) Compute normalized weights
    r2s = np.array([tabpfn_r2, tree_r2, tabnet_r2])
    weights = r2s / r2s.sum()
    print("[INFO] Ensemble weights:", dict(
        TabPFN=weights[0],
        Tree=weights[1],
        TabNet=weights[2]
    ))

    # 5) Load models
    models = [
        load_model(Path(tabpfn_path)),
        load_model(Path(tree_path)),
        load_model(Path(tabnet_path))
    ]

    # 6) Build and save ensemble
    ensemble = WeightedEnsemble(models, weights)
    final_path = output_dir / "final_model.pkl"
    with open(final_path, "wb") as f:
        pickle.dump(ensemble, f)
    print(f"[INFO] Saved weighted ensemble to {final_path}")
    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train multiple models and build weighted ensemble"
    )
    parser.add_argument(
        "-d", "--dataset", required=True,
        help="Path to dataset root folder containing fold files"
    )
    parser.add_argument(
        "-o", "--out-dir", dest="out_dir", required=True,
        help="Directory where models and final_model.pkl will be saved"
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=1,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    dataset = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_and_ensemble(dataset, out_dir, seed=args.seed)

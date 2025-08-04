#!/usr/bin/env python3
import argparse
import pickle
import logging
import sys
from pathlib import Path
from pytorch_tabnet.tab_model import TabNetRegressor
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
    Load a model by file extension (.pkl/.joblib via joblib,
    .pt/.pth via torch, TabNet .zip via TabNetRegressor.load_model).
    """
    # If they passed a directory, look inside for a .zip
    if path.is_dir():
        zips = list(path.rglob("*.zip"))
        if not zips:
            raise FileNotFoundError(f"No TabNet .zip in {path}")
        path = zips[0]   # take the first one you find

    ext = path.suffix.lower()
    if ext in {'.pkl', '.joblib'}:
        return joblib.load(path)
    if ext in {'.pt', '.pth'}:
        return torch.load(path, map_location='cpu')
    if ext == '.zip':
        # load a PyTorch-TabNet model
        model = TabNetRegressor()
        model.load_model(str(path))
        return model

    # fallback
    return joblib.load(path)


def train_and_ensemble(dataset: Path, output_dir: Path, seed: int = 1):
    logger.info("Using device: %s", _device())
    # 3) TabPFN model
    logger.info("Training TabPFN model...")
    tabpfn_path, tabpfn_r2 = train_bootstrap(
        str(dataset), output_dir=output_dir
    )
    logger.info("TabPFN →      %s (R²=%.4f)", tabpfn_path, tabpfn_r2)

    # 1) Tree-based model
    logger.info("Training tree-based model...")
    tree_path, tree_r2 = tree_based_methods_model(
        str(dataset), str(output_dir)
    )
    logger.info("Tree-based → %s (R²=%.4f)", tree_path, tree_r2)

    # 2) TabNet model
    logger.info("Training TabNet model...")
    tabnet_dir = output_dir / "tabnet"
    tabnet_dir.mkdir(parents=True, exist_ok=True)
    tabnet_path, tabnet_r2 = tabnet_model(
        str(dataset), str(tabnet_dir)
    )
    logger.info("TabNet →      %s (mean R²=%.4f)", tabnet_path, tabnet_r2)

    # 4) Compute normalized weights
    r2s = np.array([tabpfn_r2, tree_r2, tabnet_r2])
    weights = r2s / r2s.sum()
    logger.info(
        "Ensemble weights: TabPFN=%.3f, Tree=%.3f, TabNet=%.3f",
        weights[0], weights[1], weights[2]
    )

    # 5) Load models
    logger.info("Loading individual models into memory...")
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
    logger.info("Saved weighted ensemble to %s", final_path)

    return final_path


def train_best_model(dataset: Path, output_dir: Path, seed: int = 1):
    logger.info("Using device: %s", _device())

    # Train all three models (same as before)
    logger.info("Training TabPFN model...")
    tabpfn_path, tabpfn_r2 = train_bootstrap(
        str(dataset), output_dir=output_dir
    )
    logger.info("TabPFN →      %s (R²=%.4f)", tabpfn_path, tabpfn_r2)

    logger.info("Training tree-based model...")
    tree_path, tree_r2 = tree_based_methods_model(
        str(dataset), str(output_dir)
    )
    logger.info("Tree-based → %s (R²=%.4f)", tree_path, tree_r2)

    logger.info("Training TabNet model...")
    tabnet_dir = output_dir / "tabnet"
    tabnet_dir.mkdir(parents=True, exist_ok=True)
    tabnet_path, tabnet_r2 = tabnet_model(
        str(dataset), str(tabnet_dir)
    )
    logger.info("TabNet →      %s (mean R²=%.4f)", tabnet_path, tabnet_r2)

    # Identify best model by R²
    r2s = [tabpfn_r2, tree_r2, tabnet_r2]
    paths = [tabpfn_path, tree_path, tabnet_path]
    names = ["TabPFN", "Tree-based", "TabNet"]
    best_idx = int(np.argmax(r2s))
    best_name = names[best_idx]
    best_path = paths[best_idx]
    best_r2 = r2s[best_idx]

    logger.info("Best model is %s with R²=%.4f; saving as final model.", best_name, best_r2)

    # Load and re-save the single best model as final_model.pkl (keeping same interface)
    best_model = load_model(Path(best_path))
    # wrap in existing WeightedEnsemble so interface (weights, model_names reconstruction, feature engineering) stays consistent
    ensemble_obj = WeightedEnsemble([best_model], [1.0])
    final_path = output_dir / "final_model.pkl"
    with open(final_path, "wb") as f:
        pickle.dump(ensemble_obj, f)

    logger.info("Saved best single model to %s", final_path)

    return final_path


if __name__ == "__main__":
    # configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Train three models, pick the best by R², and save it as the final model"
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

    try:
        train_best_model(dataset, out_dir, seed=args.seed)
    except Exception as e:
        logger.exception("Training pipeline failed:")
        sys.exit(1)

#!/usr/bin/env python3
"""
Script to load a weighted ensemble model and generate predictions on X_test via get_test_data,
applying feature engineering only for the tree-based and TabNet submodels.
Emits an INFO log every 100th sample showing each model's prediction and the final ensemble.
"""
import sys
from pathlib import Path
import argparse
import logging
import joblib
import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from catboost.core import CatBoostRegressor

# ── ADD project’s src/ directory to PYTHONPATH ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import get_test_data
from models.tree_based_methods.auto_ml_pipeline_project.auto_ml_pipeline.feature_engineering import engineer_features

# ── LOGGER SETUP ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# ── DEVICE SETUP ───────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")


# ── HELPER TO LOAD ALL TABNET FOLDS + THEIR SCALERS ────────────────────────────
class TabNetFoldEnsemble:
    """
    Wraps K-fold TabNet models plus their StandardScalers so that predict()
    returns inverse-transformed outputs on the same scale as the original target.
    """
    def __init__(self, tabnet_dir: Path):
        import torch  # ensure torch is available here
        fold_dirs = sorted(
            d for d in tabnet_dir.iterdir()
            if d.is_dir() and d.name.startswith("fold")
        )
        self.models = []
        self.scalers = []
        for fd in fold_dirs:
            mz = fd / f"tabnet_{fd.name}.zip"
            if not mz.exists():
                raise FileNotFoundError(f"Expected TabNet zip at {mz}")
            m = TabNetRegressor()
            # load onto GPU if available
            m.load_model(str(mz), device_name='cuda' if torch.cuda.is_available() else 'cpu')
            self.models.append(m)

            sp = fd / f"scaler_{fd.name}.pkl"
            if not sp.exists():
                raise FileNotFoundError(f"Expected scaler pickle at {sp}")
            s = joblib.load(sp)
            self.scalers.append(s)

    def predict(self, X_fe: np.ndarray) -> np.ndarray:
        preds = []
        for model, scaler in zip(self.models, self.scalers):
            p_scaled = model.predict(X_fe).ravel()
            p = scaler.inverse_transform(p_scaled.reshape(-1, 1)).ravel()
            preds.append(p)
        return np.mean(preds, axis=0)


# ── WEIGHTED ENSEMBLE ──────────────────────────────────────────────────────────
class WeightedEnsemble:
    """
    Wraps models supporting .predict(...) to return a weighted average of predictions.
    """
    def __init__(self, models, weights):
        self.models = models
        self.model_names = []
        for m in models:
            if isinstance(m, CatBoostRegressor):
                self.model_names.append("CatBoost")
            elif isinstance(m, (TabNetRegressor, TabNetFoldEnsemble)):
                self.model_names.append("TabNet")
            else:
                self.model_names.append("TabPFN")
        w = np.array(weights, dtype=float)
        self.weights = w / w.sum()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # engineer features once (for CatBoost & TabNet)
        X_fe, _ = engineer_features(X, X.copy())
        X_np = X_fe.values
        n = X.shape[0]

        pred_list = []
        for m in self.models:
            if isinstance(m, (CatBoostRegressor, TabNetRegressor, TabNetFoldEnsemble)):
                inp = X_np
            else:
                # assume PyTorch-based model
                tensor = torch.from_numpy(X.values.astype(np.float32)).to(DEVICE)
                inp = tensor
            raw = m.predict(inp) if not hasattr(m, "to") else m.predict(inp)
            # if a torch model returned a tensor, convert to numpy
            if isinstance(raw, torch.Tensor):
                arr = raw.detach().cpu().numpy().ravel()
            else:
                arr = np.asarray(raw).ravel()
            if arr.shape[0] != n:
                raise ValueError(f"{type(m).__name__} returned {arr.shape[0]} preds; expected {n}")
            pred_list.append(arr)

        mat = np.vstack(pred_list)          # (n_models, n_samples)
        ens = self.weights.dot(mat)         # (n_samples,)

        for i in range(0, n, 100):
            parts = [f"{name}={mat[j,i]:.4f}" for j, name in enumerate(self.model_names)]
            parts.append(f"Ensemble={ens[i]:.4f}")
            logger.info(f"Sample {i}: " + ", ".join(parts))
        return ens


# ── MODEL LOADING ─────────────────────────────────────────────────────────────
def load_model(path: Path):
    """
    Given a Path, either:
      - return TabNetFoldEnsemble(path) if it's the tabnet/ directory
      - load joblibs, torch .pt/.pth/.pkl, or single .zip onto the correct device
    """
    if path.is_dir() and any(path.glob("fold*/tabnet_fold*.zip")):
        return TabNetFoldEnsemble(path)

    ext = path.suffix.lower()

    if ext in {'.pkl', '.pt', '.pth'}:
        # torch checkpoint — load and map to GPU if available
        model = torch.load(str(path), map_location=DEVICE, weights_only=False)
        # if model supports .to(), move it explicitly
        if hasattr(model, "to"):
            model.to(DEVICE)
        return model

    if ext == '.joblib':
        return joblib.load(path)

    if ext == '.zip':
        m = TabNetRegressor()
        m.load_model(str(path), device_name='cuda' if torch.cuda.is_available() else 'cpu')
        return m

    raise ValueError(f"Cannot load model from {path!r}")


def save_predictions(y_pred: np.ndarray, out_path: Path):
    pd.DataFrame({'y_pred': y_pred}).to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Predict with weighted ensemble")
    parser.add_argument("-d", "--dataset",   required=True, help="Path to dataset root")
    parser.add_argument("-m", "--model-file",required=True, help="Path to final_model.pkl")
    parser.add_argument("-f", "--fold",      type=int, default=None,
                        help="Fold number (if using simulated folds)")
    parser.add_argument("-o", "--output",    required=True, help="CSV path for y_pred")
    args = parser.parse_args()

    logger.info(f"Loading X_test (fold={args.fold}) from {args.dataset}")
    X_test = get_test_data(str(args.dataset), fold=args.fold)

    logger.info(f"Loading ensemble from {args.model_file}")
    ensemble = load_model(Path(args.model_file))

    # ensure we have model_names and a consistent interface for single models
    if not hasattr(ensemble, "models"):
        # wrap single model so the rest of the code can treat it uniformly
        class SingleWrapper:
            def __init__(self, model):
                self.models = [model]
                if isinstance(model, CatBoostRegressor):
                    self.model_names = ["CatBoost"]
                elif isinstance(model, (TabNetRegressor,)):
                    self.model_names = ["TabNet"]
                else:
                    self.model_names = ["TabPFN"]
                self.weights = np.array([1.0])

            def predict(self, X):
                return self.models[0].predict(X)

        ensemble = SingleWrapper(ensemble)
    elif not hasattr(ensemble, "model_names"):
        # legacy ensemble missing names: reconstruct
        ensemble.model_names = WeightedEnsemble(ensemble.models, []).model_names

    logger.info("Computing predictions...")
    y_pred = ensemble.predict(X_test)

    logger.info(f"Saving predictions to {args.output}")
    save_predictions(y_pred, Path(args.output))
    print(f"Saved predictions to: {args.output}", file=sys.stdout)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────────
# Simple wrapper that averages predictions from multiple TabPFNRegressor
# models saved as pickles and exports a single ensemble file.
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class EnsemblePFN:
    """Average predictions of a list of fitted TabPFN models."""

    def __init__(self, model_paths: list[Path]):
        self.models = [pickle.load(open(p, "rb")) for p in model_paths]

    # --------------------------------------------------------------------- #
    def predict(self, X):
        if not self.models:
            raise ValueError("EnsemblePFN.models is empty – nothing to predict with.")
        preds = np.stack([m.predict(X) for m in self.models], axis=0)
        return preds.mean(axis=0)
    # --------------------------------------------------------------------- #


# ──────────────────────────────────────────────────────────────────────────────
# Helper: discover fold files in a directory and export an ensemble.
# ──────────────────────────────────────────────────────────────────────────────
def build_ensemble(
    model_dir: Path,
    pattern: str = "fold*.pkl",
    ensemble_name: str | None = None,
) -> Path:
    """
    Search `model_dir` for fold pickles, build an EnsemblePFN, save it.

    Parameters
    ----------
    model_dir     Directory that contains the fold pickles.
    pattern       Glob pattern to match fold files (default: 'fold*.pkl').
    ensemble_name If given, the ensemble is saved as '<ensemble_name>.pkl'.
                  Otherwise it is saved as 'ensemble.pkl'.

    Returns
    -------
    Path to the saved ensemble pickle.
    """
    paths = sorted(model_dir.glob(pattern))
    if not paths:
        raise RuntimeError(f"No fold models matching '{pattern}' found in {model_dir}")

    ensemble = EnsemblePFN(paths)
    file_name = f"{ensemble_name}.pkl" if ensemble_name else "ensemble.pkl"
    out_path = model_dir / file_name

    with out_path.open("wb") as f:
        pickle.dump(ensemble, f)

    print(f"[Ensemble] Saved → {out_path}")
    return out_path

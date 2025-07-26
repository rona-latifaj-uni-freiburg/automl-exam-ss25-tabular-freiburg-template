#!/usr/bin/env python3
# bootstrap_ensemble.py

import pickle
from pathlib import Path

import numpy as np


class EnsemblePFN:
    """
    Averages predictions from multiple pickled TabPFNRegressor models.
    """

    def __init__(self, model_paths: list[Path]):
        self.models = [pickle.load(open(p, "rb")) for p in model_paths]
        if not self.models:
            raise ValueError("No models found to ensemble.")

    def predict(self, X):
        # stack: (n_models, n_samples, ...) → mean over axis=0
        preds = np.stack([m.predict(X) for m in self.models], axis=0)
        return preds.mean(axis=0)

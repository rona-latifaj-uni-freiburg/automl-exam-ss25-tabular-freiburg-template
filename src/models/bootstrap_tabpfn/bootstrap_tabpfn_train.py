#!/usr/bin/env python3
# bootstrap_tabpfn_train.py

import argparse
import logging
import pickle
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score
from tabpfn import TabPFNRegressor
from torch.utils.tensorboard import SummaryWriter

import optuna

from data import simulate_exam_dataset, get_test_data
from .bootstrap_ensemble import EnsemblePFN

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_bootstrap(
    dataset: str,
    output_dir: Path,
    n_bootstrap: int = 10,
    sample_frac: float = 0.8,
    seed: int = 0,
) -> float:
    start_time = time.time()
    device = _device()
    ds_name = Path(dataset).name
    outdir = output_dir / ds_name
    outdir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=str(output_dir / "runs" / ts))
    logger.info(f"Bootstrap training → dataset={ds_name}, device={device}")

    X_full, y_full, _ = simulate_exam_dataset(dataset)
    n_samples = len(X_full)
    sample_size = int(n_samples * sample_frac)
    model_paths: list[Path] = []

    oob_preds = defaultdict(list)

    for i in range(n_bootstrap):
        np.random.seed(seed + i)
        torch.manual_seed(seed + i)

        idx = np.random.choice(n_samples, size=sample_size, replace=True)
        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[np.unique(idx)] = False

        X_bs = X_full.iloc[idx]
        y_bs = y_full.iloc[idx].values.ravel()

        logger.info(f"[{i+1}/{n_bootstrap}] bootstrap sample size={sample_size}")
        agent = TabPFNRegressor(device=device)
        agent.fit(X_bs, y_bs)

        if np.any(oob_mask):
            X_oob = X_full[oob_mask]
            pred_oob = agent.predict(X_oob)
            for row_idx, pred_val in zip(np.where(oob_mask)[0], pred_oob):
                oob_preds[row_idx].append(pred_val)

        out_path = outdir / f"bootstrap_{i+1}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(agent, f)
        model_paths.append(out_path)

        del agent
        if device == "cuda":
            torch.cuda.empty_cache()

    final_preds = np.zeros(n_samples)
    used_mask = np.zeros(n_samples, dtype=bool)

    for row_idx, preds in oob_preds.items():
        final_preds[row_idx] = np.mean(preds)
        used_mask[row_idx] = True

    y_true = y_full.values.ravel()
    y_used = y_true[used_mask]
    y_pred = final_preds[used_mask]

    oob_r2 = r2_score(y_used, y_pred)
    logger.info(f"📊 OOB R² Score = {oob_r2:.5f}")
    writer.add_scalar("R2_score/OOB", oob_r2, 0)

    ensemble = EnsemblePFN(model_paths)
    ens_path = outdir / "ensemble.pkl"
    with open(ens_path, "wb") as f:
        pickle.dump(ensemble, f)
    logger.info(f"Saved ensemble → {ens_path}")

    writer.close()
    logger.info(f"Total time: {time.time() - start_time:.1f}s")
    return oob_r2


def objective(trial):
    n_bootstrap = trial.suggest_int("n_bootstrap", 5, 15)
    sample_frac = trial.suggest_float("sample_frac", 0.6, 1.0)
    logger.info(f"🔍 Trying: n_bootstrap={n_bootstrap}, sample_frac={sample_frac:.2f}")

    try:
        score = train_bootstrap(
            dataset=args.dataset,
            output_dir=Path(args.out_dir) / "optuna_trials",
            n_bootstrap=n_bootstrap,
            sample_frac=sample_frac,
            seed=args.seed,
        )
        return score
    except Exception as e:
        logger.error(f"Trial failed: {e}")
        return -np.inf


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Bootstrap-bagging TabPFN with optional Optuna tuning")
    p.add_argument("-d", "--dataset", required=True, help="dataset root folder")
    p.add_argument("-o", "--out-dir", default="models", help="output directory")
    p.add_argument("--n-bootstrap", type=int, default=10, help="number of bootstrap samples")
    p.add_argument("--sample-frac", type=float, default=0.8, help="fraction of rows/sample")
    p.add_argument("--seed", type=int, default=0, help="random seed")
    p.add_argument("--optuna", action="store_true", help="Use Bayesian Optimization for tuning")

    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.optuna:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)
        logger.info(f"🏆 Best Params: {study.best_params}, R²={study.best_value:.5f}")
    else:
        r2 = train_bootstrap(
            dataset=args.dataset,
            output_dir=out,
            n_bootstrap=args.n_bootstrap,
            sample_frac=args.sample_frac,
            seed=args.seed,
        )
        logger.info(f"📊 Final OOB R²: {r2:.5f}")

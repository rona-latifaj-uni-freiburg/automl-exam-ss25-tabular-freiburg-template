#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────────
# Train TabPFNRegressor on every fold of a dataset and save:
#   • one model per fold         →  <output_dir>/<dataset>/fold<k>.pkl
#   • an ensemble                →  <output_dir>/<dataset>/ensemble.pkl
#   • TensorBoard logs           →  <output_dir>/runs/<timestamp>/
# Also logs ensemble‑level metrics and system stats to TensorBoard.
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from tabpfn import TabPFNRegressor
from torch.utils.tensorboard import SummaryWriter

from data import get_available_folds, load_fold
from .tabpfn_ensemble import build_ensemble, EnsemblePFN

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─── Training function ────────────────────────────────────────────────────────
def train_folds(dataset: str, output_dir: Path, seed: int = 0):
    """
    dataset
        Path to dataset root (contains fold parquet files).
    output_dir
        Root directory; every dataset gets its own sub‑folder here.
    seed
        RNG seed for reproducibility.
    """
    start_time = time.time()
    device = _device()
    folds = get_available_folds(dataset)
    dataset_name = Path(dataset).name

    # All artefacts live in <output_dir>/<dataset_name>/
    dataset_outdir = output_dir / dataset_name
    dataset_outdir.mkdir(parents=True, exist_ok=True)

    scores, model_paths = [], []
    X_test_parts, y_test_parts = [], []         # for later ensemble eval

    # TensorBoard logs under <output_dir>/runs/<timestamp>/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_log_dir = output_dir / "runs" / timestamp
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    logger.info(f"TensorBoard logs → {tb_log_dir}")

    # ---------- fold loop ----------------------------------------------------
    for fold in folds:
        try:
            X_tr, X_te, y_tr, y_te = load_fold(dataset, fold)
            logger.info(f"Fold {fold}: {len(X_tr)} train rows  •  device={device}")

            torch.manual_seed(seed)
            np.random.seed(seed)

            agent = TabPFNRegressor(device=device)
            agent.fit(X_tr, y_tr.values.ravel())

            y_pred = agent.predict(X_te)
            score = r2_score(y_te, y_pred)
            logger.info(f"Fold {fold}  R² = {score:.4f}")
            writer.add_scalar("R2_score/fold", score, fold)

            # save fold model
            fold_path = dataset_outdir / f"fold{fold}.pkl"
            with fold_path.open("wb") as f:
                pickle.dump(agent, f)
            logger.info(f"Saved fold model → {fold_path}")

            scores.append(score)
            model_paths.append(fold_path)
            X_test_parts.append(X_te)
            y_test_parts.append(y_te)

            # free GPU memory
            del agent
            if device == "cuda":
                torch.cuda.empty_cache()

        except Exception:
            logger.exception(f"Fold {fold} failed")
            if device == "cuda":
                torch.cuda.empty_cache()

    # ---------- summary & ensemble ------------------------------------------
    if scores:
        mean, std = np.mean(scores), np.std(scores)
        logger.info(f"Mean R² over {len(scores)} folds: {mean:.4f} ± {std:.4f}")
        writer.add_scalar("R2_score/mean", mean, 0)
        writer.add_scalar("R2_score/std", std, 0)

        # Build ensemble and evaluate on concatenated test sets
        ens_path = build_ensemble(dataset_outdir, pattern="fold*.pkl")
        ensemble = pickle.load(open(ens_path, "rb"))

        X_test_full = pd.concat(X_test_parts, ignore_index=True)
        y_test_full = pd.concat(y_test_parts, ignore_index=True).squeeze("columns")
        y_pred_full = ensemble.predict(X_test_full).squeeze()

        r2_ens = r2_score(y_test_full, y_pred_full)
        writer.add_scalar("R2_score/ensemble", r2_ens, 0)
        writer.add_histogram("residuals/ensemble", y_pred_full - y_test_full.values, 0)
        logger.info(f"Ensemble  R² = {r2_ens:.4f}")

    else:
        logger.warning("No successful folds; nothing to log for ensemble.")

    # ---------- system‑level stats ------------------------------------------
    elapsed = time.time() - start_time
    writer.add_scalar("time_sec/total", elapsed, 0)
    if device == "cuda":
        gpu_mem = torch.cuda.max_memory_allocated()
        writer.add_scalar("GPU_mem_bytes/max_allocated", gpu_mem, 0)

    writer.add_text("hparams", f"seed: {seed}\ndevice: {device}", 0)
    writer.close()

    return model_paths, scores


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="TabPFN fold-wise training")
    p.add_argument("-d", "--dataset", required=True, help="dataset root folder")
    p.add_argument(
        "-o", "--out-dir", default="models",
        help="output directory for fold & ensemble pickles"
    )
    p.add_argument("--seed", type=int, default=0, help="random seed")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_folds(dataset=args.dataset, output_dir=out_dir, seed=args.seed)

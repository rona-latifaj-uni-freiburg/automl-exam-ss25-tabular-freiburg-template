#!/usr/bin/env python3
# bootstrap_tabpfn_train.py

import argparse
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tabpfn import TabPFNRegressor
from torch.utils.tensorboard import SummaryWriter

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
):
    """
    1) Picks one random fold via simulate_exam_dataset()
    2) Creates `n_bootstrap` samples (with replacement) of size ~sample_frac
    3) Trains one TabPFN per sample and saves each to:
         <output_dir>/<dataset>/bootstrap_<i>.pkl
    4) Builds & saves an EnsemblePFN averaging all bootstraps
    """
    start_time = time.time()
    device = _device()
    ds_name = Path(dataset).name
    outdir = output_dir / ds_name
    outdir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=str(output_dir / "runs" / ts))
    logger.info(f"Bootstrap training → dataset={ds_name}, device={device}")

    # 1) load one random fold
    X_full, y_full = simulate_exam_dataset(dataset)
    X_test = get_test_data(dataset)

    n_samples = len(X_full)
    sample_size = int(n_samples * sample_frac)
    model_paths: list[Path] = []

    for i in range(n_bootstrap):
        # deterministic per-bootstrp
        np.random.seed(seed + i)
        torch.manual_seed(seed + i)

        idx = np.random.choice(n_samples, size=sample_size, replace=True)
        X_bs = X_full.iloc[idx]
        y_bs = y_full.iloc[idx].values.ravel()

        logger.info(f"[{i+1}/{n_bootstrap}] bootstrap sample size={sample_size}")
        agent = TabPFNRegressor(device=device)
        agent.fit(X_bs, y_bs)

        out_path = outdir / f"bootstrap_{i+1}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(agent, f)
        logger.info(f"Saved model → {out_path}")
        model_paths.append(out_path)

        # free GPU mem
        del agent
        if device == "cuda":
            torch.cuda.empty_cache()

    # build & save ensemble
    ensemble = EnsemblePFN(model_paths)
    ens_path = outdir / "ensemble.pkl"
    with open(ens_path, "wb") as f:
        pickle.dump(ensemble, f)
    logger.info(f"Saved ensemble → {ens_path}")

    # wrap up
    writer.close()
    logger.info(f"Total time: {time.time()-start_time:.1f}s")
    return model_paths, ens_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Bootstrap-bagging TabPFN on a single random fold"
    )
    p.add_argument("-d", "--dataset",      required=True, help="dataset root folder")
    p.add_argument("-o", "--out-dir",      default="models", help="where to save models")
    p.add_argument("--n-bootstrap", type=int,   default=10,    help="number of bootstrap samples")
    p.add_argument("--sample-frac", type=float, default=0.8,   help="fraction of rows/sample")
    p.add_argument("--seed",        type=int,   default=0,     help="random seed")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_bootstrap(
        dataset=args.dataset,
        output_dir=out,
        n_bootstrap=args.n_bootstrap,
        sample_frac=args.sample_frac,
        seed=args.seed,
    )

#!/usr/bin/env python3
# main.py

import argparse
from pathlib import Path

from bootstrap_tabpfn_train import train_bootstrap

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run TabPFN bootstrap-bagging then save an ensemble"
    )
    parser.add_argument(
        "-d", "--dataset", required=True, help="root folder containing folds"
    )
    parser.add_argument(
        "-o", "--out-dir", default="models", help="where to save models"
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=10, help="number of bootstrap samples"
    )
    parser.add_argument(
        "--sample-frac", type=float, default=0.8, help="fraction of rows/sample"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_bootstrap(
        dataset=args.dataset,
        output_dir=out,
        n_bootstrap=args.n_bootstrap,
        sample_frac=args.sample_frac,
        seed=args.seed,
    )

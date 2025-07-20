# src/models/tabpfn/__main__.py

import argparse
from pathlib import Path

from .tabpfn_train import train_folds
from .tabpfn_ensemble import build_ensemble


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run TabPFN training then ensemble"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        help="root folder containing folds",
    )
    parser.add_argument(
        "-o",
        "--model-dir",
        default="models",
        help="where to save fold and ensemble models",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for all folds",
    )
    args = parser.parse_args()

    # 1) Train each fold (uses TabPFN's default ensemble size)
    train_folds(args.dataset, Path(args.model_dir), seed=args.seed)

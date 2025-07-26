#!/usr/bin/env python3
# __main__.py

import argparse
from pathlib import Path
import optuna
import logging

from .bootstrap_tabpfn_train import train_bootstrap, objective

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run TabPFN bootstrap-bagging with optional Bayesian Optimization"
    )
    parser.add_argument("-d", "--dataset",      required=True,
                        help="Root folder containing folds")
    parser.add_argument("-o", "--out-dir",      default="models",
                        help="Where to save fold and ensemble models")
    parser.add_argument("--n-bootstrap",        type=int,   default=10,
                        help="Number of bootstrap samples (manual mode)")
    parser.add_argument("--sample-frac",        type=float, default=0.8,
                        help="Fraction of rows per sample (manual mode)")
    parser.add_argument("--seed",               type=int,   default=0,
                        help="Random seed")
    parser.add_argument("--fold",               type=int,   default=None,
                        help="Which fold to use (overrides random)")
    parser.add_argument("--optuna",             action="store_true",
                        help="Use Bayesian Optimization to tune parameters")
    parser.add_argument("--n-trials",           type=int,   default=10,
                        help="Number of Optuna trials when --optuna is set")

    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.optuna:
        # create study with pruning etc. (mirror what you have in train file)
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=0,
            interval_steps=1
        )
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=args.seed),
            pruner=pruner
        )

        # Optimize for args.n_trials
        study.optimize(lambda t: objective(t, args), n_trials=args.n_trials)

        print(f"🏆 Best parameters: {study.best_params} → R²: {study.best_value:.5f}")

        # Retrain final ensemble with those best params
        best = study.best_params
        final_r2 = train_bootstrap(
            dataset=args.dataset,
            output_dir=out,
            n_bootstrap=best["n_bootstrap"],
            sample_frac=best["sample_frac"],
            seed=args.seed,
            fold=args.fold,
        )
        print(f"📦 Final OOB R² with best params → {final_r2:.5f}")

    else:
        # manual mode
        r2 = train_bootstrap(
            dataset=args.dataset,
            output_dir=out,
            n_bootstrap=args.n_bootstrap,
            sample_frac=args.sample_frac,
            seed=args.seed,
            fold=args.fold,
        )
        print(f"📊 Final OOB R²: {r2:.5f}")

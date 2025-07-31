# src/models/tabnet/tabnet_train.py

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from pytorch_tabnet.tab_model import TabNetRegressor
import optuna
from sklearn.preprocessing import StandardScaler
import joblib

#-----------------------------
# Utility: Function to convert data to float32 numpy arrays
#-----------------------------
def to_numpy_float32(x):
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.astype(np.float32).values
    return np.array(x).astype(np.float32)


#-----------------------------
# Objective function for Optuna hyperparameter optimization
#-----------------------------
def objective(trial, X_train, y_train, X_val, y_val):
    params = {
            'n_d': trial.suggest_int('n_d', 8, 64),
            'n_a': trial.suggest_int('n_a', 8, 64),
            'n_steps': trial.suggest_int('n_steps', 3, 10),
            'gamma': trial.suggest_float('gamma', 1.0, 2.0),
            'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-5, 1e-2, log=True),
            'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        }

    clf = TabNetRegressor(
        n_d=params['n_d'],
        n_a=params['n_a'],
        n_steps=params['n_steps'],
        gamma=params['gamma'],
        lambda_sparse=params['lambda_sparse'],
        optimizer_params=dict(lr=params['lr'], weight_decay=params['weight_decay']),
        mask_type='entmax',
        verbose=0,
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=['rmse'],
        max_epochs=100,
        patience=20,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,     
    )

    preds = clf.predict(X_val).ravel()
    r2 = r2_score(y_val.ravel(), preds)
    return r2

#-----------------------------
# Function to train TabNet with KFold and Optuna hyperparameter optimization
#-----------------------------
def train_fold_with_optuna(X_tr, y_tr, X_val, y_val, fold_idx, output_dir, n_trials=20):
    print(f"Fold {fold_idx} started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    X_train_np = to_numpy_float32(X_tr)
    X_val_np = to_numpy_float32(X_val)

    # scale the target variable for better optimization
    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(np.asarray(to_numpy_float32(y_tr)).reshape(-1, 1))
    y_val_scaled = scaler.transform(np.asarray(to_numpy_float32(y_val)).reshape(-1, 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = output_dir / f"scaler_fold{fold_idx}.pkl"
    joblib.dump(scaler, scaler_path)

    # Optimize hyperparameters using scaled targets
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train_np, y_train_scaled, X_val_np, y_val_scaled), n_trials=n_trials, n_jobs=1, catch=(Exception,))

    print(f"Best params for fold {fold_idx}: {study.best_params}")
    print(f"Best R² for fold {fold_idx}: {study.best_value:.4f}")

    # Train final model with best params on scaled targets
    best_params = study.best_params
    clf = TabNetRegressor(
        n_d=best_params['n_d'],
        n_a=best_params['n_a'],
        n_steps=best_params['n_steps'],
        gamma=best_params['gamma'],
        lambda_sparse=best_params['lambda_sparse'],
        optimizer_params=dict(
            lr=best_params['lr'],
            weight_decay=best_params['weight_decay']
        ),
        mask_type='entmax',
        verbose=0,
    )
    clf.fit(
        X_train_np, y_train_scaled,
        eval_set=[(X_val_np, y_val_scaled)],
        eval_metric=['rmse'],
        max_epochs=150,
        patience=30,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
    )

    # caluclate the r^2 score by predicting on the validation set
    val_preds_scaled = clf.predict(X_val_np).ravel()
    val_preds = scaler.inverse_transform(val_preds_scaled.reshape(-1, 1)).ravel()
    y_val_orig = scaler.inverse_transform(y_val_scaled).ravel()

    val_r2 = r2_score(y_val_orig, val_preds)
    print(f"Fold {fold_idx} final validation R²: {val_r2:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"tabnet_fold{fold_idx}"
    clf.save_model(str(model_path))
    print(f"Saved fold {fold_idx} model to {model_path}")

    return val_r2, study.best_params

#-----------------------------
# Main function to run TabNet training with KFold and Optuna
#-----------------------------
def main(dataset_dir, output_dir, n_splits=5, n_trials=20):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    # Load full train data
    X = pd.read_parquet(dataset_dir / "X_train.parquet")
    y = pd.read_parquet(dataset_dir / "y_train.parquet")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2_scores = []

    for fold_idx, (train_index, val_index) in enumerate(kf.split(X), start=1):
        X_tr, X_val = X.iloc[train_index], X.iloc[val_index]
        y_tr, y_val = y.iloc[train_index], y.iloc[val_index]

        val_r2, best_params = train_fold_with_optuna(X_tr, y_tr, X_val, y_val, fold_idx, output_dir, n_trials)
        r2_scores.append(val_r2)

    print(f"\nMean R² across {n_splits} folds: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")

#-----------------------------
# CLI             
#-----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_trials", type=int, default=30)
    args = parser.parse_args()

    main(args.dataset_dir, args.output_dir, args.n_splits, args.n_trials)

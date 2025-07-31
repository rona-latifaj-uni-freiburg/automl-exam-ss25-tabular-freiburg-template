import argparse
import subprocess
import os
import numpy as np
from typing import Tuple
from .tabnet_train import main as _train_main, train_fold_with_optuna
from .tabnet_ensembling import main as _ensemble_main
from src.data.load import load_only_train_for_dataset
from pathlib import Path
from sklearn.model_selection import KFold

def tabnet_model(
    dataset_path: str,
    output_dir: str,
    n_splits: int = 10,
    n_trials: int = 20,
    seed: int = 42
) -> Tuple[str, float]:
    """
    Train TabNet via K-Fold + Optuna on merged splits, reporting progress via prints.

    Args:
        dataset_path: path to folder with numbered train splits
        output_dir: directory to save fold models
        n_splits: number of CV folds
        n_trials: Optuna trials per fold
        seed: random seed for reproducibility

    Returns:
        model_dir: path to output_dir
        mean_r2: mean validation R² across folds
    """
    # prepare output
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output directory '{output_dir}' ready.")

    # load & merge all train splits
    ds = Path(dataset_path).name
    X_all, y_all = load_only_train_for_dataset(ds)
    print(f"[INFO] Loaded merged train data: {X_all.shape[0]} rows")

    # K-Fold CV
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    r2_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_all), start=1):
        print(f"[INFO] Fold {fold_idx}/{n_splits} starting...")
        X_tr = X_all.iloc[train_idx]
        X_val = X_all.iloc[val_idx]
        y_tr = y_all.iloc[train_idx]
        y_val = y_all.iloc[val_idx]

        fold_dir = Path(output_dir) / f"fold{fold_idx}"
        os.makedirs(fold_dir, exist_ok=True)
        val_r2, _ = train_fold_with_optuna(
            X_tr, y_tr,
            X_val, y_val,
            fold_idx=fold_idx,
            output_dir=fold_dir,
            n_trials=n_trials
        )
        r2_scores.append(val_r2)
        print(f"[INFO] Fold {fold_idx} done. R²={val_r2:.4f}")

    mean_r2 = float(np.mean(r2_scores))
    std_r2 = float(np.std(r2_scores))
    print(f"[INFO] TabNet CV complete — mean R² = {mean_r2:.4f} ± {std_r2:.4f}")

    return output_dir, mean_r2
#-----------------------------
#  Function to run TabNet training via subprocess (unchanged)
#-----------------------------
def run_training(dataset_path: str, output_dir: str):
    n_splits = 10
    n_trials = 20
    print(f"\n[+] Starting TabNet {n_splits}-Fold CV Training...")
    train_cmd = [
        "python", "src/models/tabnet/tabnet_train.py",
        "--dataset_dir", dataset_path,
        "--output_dir", output_dir,
        "--n_splits", str(n_splits),
        "--n_trials", str(n_trials)
    ]
    subprocess.run(train_cmd, check=True)
    print(f"[+] Training completed. Fold models saved to {output_dir}")

#-----------------------------
#  Function to run TabNet ensembling and prediction
#-----------------------------  
def run_prediction(dataset_path: str, model_dir: str):
    output_file = "models/exam_output/y_pred.csv"  # fixed prediction output path
    print(f"\n[+] Starting TabNet Ensemble Prediction on Test Set...")
    predict_cmd = [
        "python", "src/models/tabnet/tabnet_ensembling.py",
        "--dataset_dir", dataset_path,
        "--model_dir", model_dir,
        "--output_file", output_file
    ]
    subprocess.run(predict_cmd, check=True)
    print(f"[+] Prediction completed. Predictions saved to {output_file}")

#-----------------------------
#  CLI
#-----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline: Train TabNet with KFold + Ensemble Predict")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--output_dir", type=str, default="models/exam_output", help="Directory to save fold models")

    args = parser.parse_args()

    run_training(args.dataset_path, args.output_dir)
    run_prediction(args.dataset_path, args.output_dir)


#usage: python src/models/tabnet/tabnet_pipeline.py --dataset_path examdata/exam_dataset/1 --output_dir models/exam_output
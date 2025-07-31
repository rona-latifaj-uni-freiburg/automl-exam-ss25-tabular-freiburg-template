# final_train.py

"""
Entry point: merges **all** training splits, holds out 10% for validation,
selects & retrains the best model on 100% of the training data.
"""
import argparse
import os
from pathlib import Path
import logging

from sklearn.model_selection import train_test_split

from .config import RANDOM_SEED
# from auto_ml_pipeline.data_processing import load_all_splits
from .auto_ml_pipeline.feature_engineering import engineer_features
from .auto_ml_pipeline.model_selection import select_and_train, EnsembleModel
from .auto_ml_pipeline.hyperparameter_tuning import tune_lgbm, tune_catboost
from .auto_ml_pipeline.utils import setup_logging, save_model, report_results
from src.data.load import load_only_train_for_dataset, DATA_ROOT
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="AutoML Pipeline: final model on train-only dataset"
    )
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to root folder with subfolders 1..N containing X_train.parquet & y_train.parquet.")
    parser.add_argument("--output_dir", type=str, default="output_final",
                        help="Directory to save the final model and report.")
    return parser.parse_args()

def tree_based_methods_model(dataset_path: str, output_dir: str):
    """on
    Train and save the best tree-based model on the full training data.

    Args:
        dataset_path: Path to the root folder with subfolders containing X_train.parquet & y_train.parquet.
        output_dir: Directory where to save the final model and metrics.

    Returns:
        model_path: Path to the saved model file.
        best_r2: R² score on the held-out validation set.
    """
    # 1) prepare output and logging
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)
    logging.info("Output directory '%s' is ready and logging is configured.", output_dir)

    # 2) load and merge all train splits
    ds = Path(dataset_path).name
    X_all, y_all = load_only_train_for_dataset(ds, DATA_ROOT)
    logging.info("Merged training data: %d rows", X_all.shape[0])

    # 3) hold out 10% for validation
    X_pool, X_val, y_pool, y_val = train_test_split(
        X_all, y_all,
        test_size=0.10,
        random_state=RANDOM_SEED
    )
    logging.info("Split data into pool (%d rows) and validation (%d rows)", X_pool.shape[0], X_val.shape[0])

    # 4) feature engineering
    X_pool_fe, X_val_fe = engineer_features(X_pool, X_val)
    if X_pool_fe.shape[1] == 0:
        logging.error("No features generated after engineering on pool")
        raise RuntimeError("No features after engineering on pool")
    logging.info("Feature engineering completed: %d features", X_pool_fe.shape[1])

    # 5) model selection on pool → val
    models, metrics = select_and_train(
        X_pool_fe, y_pool,
        X_val_fe, y_val,
        output_dir=output_dir
    )
    val_metrics = {f"val_{k}": v for k, v in metrics.items()}
    report_results(val_metrics, output_dir)
    logging.info("Model selection complete. Validation metrics: %s", val_metrics)

    # 6) choose best model
    best_name = max(metrics, key=metrics.get)
    best_r2 = metrics[best_name]
    logging.info("Selected best model '%s' with validation R²=%.4f", best_name, best_r2)

    # 7) retrain best on 100% data
    logging.info("Retraining best model '%s' on full dataset", best_name)
    X_full_fe, _ = engineer_features(X_all, X_all.copy())
    if isinstance(models[best_name], EnsembleModel):
        m1, m2 = models[best_name].m1, models[best_name].m2
        m1.fit(X_full_fe, y_all)
        m2.fit(X_full_fe, y_all)
        final_model = EnsembleModel(m1, m2)
    else:
        base = models[best_name]
        final_model = base.__class__(**base.get_params())
        final_model.fit(X_full_fe, y_all)
    logging.info("Retraining completed")

    # 8) save final model
    model_filename = f"final_{best_name}.pkl"
    model_path = os.path.join(output_dir, model_filename)
    save_model(final_model, model_path)
    logging.info("Saved final model to '%s'", model_path)

    return model_path, best_r2

def main():
    args = parse_args()
    try:
        splits = sorted(os.listdir(args.dataset_path))
    except FileNotFoundError:
        raise RuntimeError(f"Dataset path not found: {args.dataset_path}")
    if not splits:
        raise RuntimeError(f"No splits found in {args.dataset_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)

    # 1) merge all train splits via load_only_train_for_dataset
    ds = Path(args.dataset_path).name
    X_all, y_all = load_only_train_for_dataset(ds, DATA_ROOT)
    print(f"DEBUG: merged train pool = {X_all.shape[0]} rows")

    # 2) hold out 10% for validation
    X_pool, X_val, y_pool, y_val = train_test_split(
        X_all, y_all,
        test_size=0.10,
        random_state=RANDOM_SEED
    )
    print(f"DEBUG: pool={X_pool.shape[0]} rows, val={X_val.shape[0]} rows")

    # 3) feature engineering
    X_pool_fe, X_val_fe = engineer_features(X_pool, X_val)
    if X_pool_fe.shape[1] == 0:
        raise RuntimeError("No features after engineering on pool")

    # 4) model selection on pool→val
    models, metrics = select_and_train(
        X_pool_fe, y_pool,
        X_val_fe, y_val,
        output_dir=args.output_dir
    )
    # after select_and_train in final_train.py
    val_metrics = {f"val_{k}": v for k, v in metrics.items()}
    report_results(val_metrics, args.output_dir)


    # 5) choose best on validation
    best = max(metrics, key=metrics.get)
    print(f"Selected model: {best} (val R² = {metrics[best]:.4f})")

    # 6) retrain best on 100% of all training data
    X_full_fe, _ = engineer_features(X_all, X_all.copy())
    if isinstance(models[best], EnsembleModel):
        m1, m2 = models[best].m1, models[best].m2
        m1.fit(X_full_fe, y_all)
        m2.fit(X_full_fe, y_all)
        final_model = EnsembleModel(m1, m2)
    else:
        base = models[best]
        final_model = base.__class__(**base.get_params())
        final_model.fit(X_full_fe, y_all)

    # 7) save & report only validation metrics
    save_model(final_model, os.path.join(args.output_dir, f"final_{best}.pkl"))
    report_results(metrics, args.output_dir)
    print(f"Validation metrics written to {os.path.join(args.output_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()

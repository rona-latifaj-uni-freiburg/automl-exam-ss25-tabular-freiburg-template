import random
import pandas as pd
from pathlib import Path
from typing import Tuple,List,Dict
import logging

DATA_ROOT = Path(__file__).resolve().parents[2]/"data"

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_available_datasets(data_root: Path = DATA_ROOT) -> List[str]:
    """
    List all dataset names under data/, e.g. ["bike_sharing_demand", "wine_quality", ...].
    """
    return sorted([p.name for p in data_root.iterdir() if p.is_dir()])

def simulate_exam_dataset(
    dataset: str,
    data_root: Path = DATA_ROOT,
    fold: int | None = None
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Selects one fold (either the one you pass, or random if you pass None)
    and returns (X_train, y_train, fold_number). Also saves the chosen fold
    globally for `get_test_data()` fallback.
    """
    global _simulated_fold
    folds = get_available_folds(dataset, data_root)

    if fold is not None:
        if fold not in folds:
            raise ValueError(f"Fold {fold} not available for dataset {dataset}.")
        _simulated_fold = fold
    else:
        _simulated_fold = random.choice(folds)

    logger.info(f"Selected fold: {_simulated_fold} for dataset: {dataset}")
    X_tr, _, y_tr, _ = load_fold(dataset, _simulated_fold, data_root)
    return X_tr, y_tr, _simulated_fold


def get_test_data(
    dataset: str,
    data_root: Path = DATA_ROOT,
    fold: int | None = None
) -> pd.DataFrame:
    """
    Returns X_test for the specified fold, or for the last fold chosen
    by `simulate_exam_dataset()` if you omit `fold`.
    """
    global _simulated_fold
    chosen = fold if fold is not None else _simulated_fold
    if chosen is None:
        raise RuntimeError("No fold set. Call simulate_exam_dataset(..., fold=...) first.")
    logger.info(f"Loading X_test for fold {chosen} of dataset: {dataset}")
    _, X_te, _, _ = load_fold(dataset, chosen, data_root)
    return X_te


def get_available_folds(dataset: str, data_root: Path = DATA_ROOT) -> List[int]:
    """
    List all fold numbers (subfolders named '1'..'10') for a given dataset.
    """
    ds_dir = data_root / dataset
    return sorted([int(p.name) for p in ds_dir.iterdir() if p.is_dir() and p.name.isdigit()])

def _build_path(dataset: str, fold: int, fname: str, data_root: Path) -> Path:
    return data_root / dataset / str(fold) / fname


def load_fold(
    dataset: str,
    fold: int,
    data_root: Path = DATA_ROOT
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load one specific fold.
    Returns: (X_train, X_test, y_train, y_test)
    """
    files = {
        "X_train": "X_train.parquet",
        "X_test":  "X_test.parquet",
        "y_train": "y_train.parquet",
        "y_test":  "y_test.parquet",
    }
    paths = {k: _build_path(dataset, fold, fn, data_root) for k, fn in files.items()}
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {k} at {p}")
    X_train = pd.read_parquet(paths["X_train"])
    X_test  = pd.read_parquet(paths["X_test"])
    y_train = pd.read_parquet(paths["y_train"])
    y_test  = pd.read_parquet(paths["y_test"])
    return X_train, X_test, y_train, y_test

def load_all_folds(
    dataset: str,
    data_root: Path = DATA_ROOT
) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Load every fold into a dict: { fold_number: (X_tr, X_te, y_tr, y_te), ... }.
    Useful if you want to loop over all folds individually.
    """
    folds = get_available_folds(dataset, data_root)
    return {fold: load_fold(dataset, fold, data_root) for fold in folds}


def load_full_train_for_specific_dataset(
    dataset: str,
    data_root: Path = DATA_ROOT
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Concatenate X_train & y_train across ALL folds into one big training set.
    Returns: (X_full, y_full)
    """
    X_parts, y_parts = [], []
    for fold in get_available_folds(dataset, data_root):
        X_tr, _, y_tr, _ = load_fold(dataset, fold, data_root)
        X_parts.append(X_tr)
        y_parts.append(y_tr)
    X_full = pd.concat(X_parts, ignore_index=True)
    y_full = pd.concat(y_parts, ignore_index=True)
    return X_full, y_full


def load_full_test_for_specific_dataset(
    dataset: str,
    data_root: Path = DATA_ROOT
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    (Optional) Concatenate X_test & y_test across ALL folds.
    Returns: (X_full_test, y_full_test)
    """
    X_parts, y_parts = [], []
    for fold in get_available_folds(dataset, data_root):
        _, X_te, _, y_te = load_fold(dataset, fold, data_root)
        X_parts.append(X_te)
        y_parts.append(y_te)
    X_full = pd.concat(X_parts, ignore_index=True)
    y_full = pd.concat(y_parts, ignore_index=True)
    return X_full, y_full


if __name__ == "__main__":
    # Smoke test
    ds = get_available_datasets()[0]
    print(f"Dataset: {ds}")
    print("Folds:", get_available_folds(ds))

    # Single fold
    X_tr, X_te, y_tr, y_te = load_fold(ds, 1)
    print("Fold 1 shapes:", X_tr.shape, X_te.shape, y_tr.shape, y_te.shape)

    # All folds dict
    all_folds = load_all_folds(ds)
    print("Loaded folds:", list(all_folds.keys()))

    # Full train
    X_full_tr, y_full_tr = load_full_train_for_specific_dataset(ds)
    print("Full train shape:", X_full_tr.shape, y_full_tr.shape)

    # Full test (if you need it)
    X_full_te, y_full_te = load_full_test_for_specific_dataset(ds)
    print("Full test shape:", X_full_te.shape, y_full_te.shape)
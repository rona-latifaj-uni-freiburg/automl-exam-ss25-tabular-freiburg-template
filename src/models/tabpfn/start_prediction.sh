#!/bin/bash

#SBATCH --job-name=tabpfn-train
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=12:10:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

###############################################################################
#                  TabPFN training / inference SLURM script                  #
#  Edit the three path variables below to suit your project layout.           #
###############################################################################

# ── user‑configurable paths --------------------------------------------------
PROJECT_ROOT="/work/dlclarge2/alidemaa-dl_lab/automl/automl-exam-ss25-tabular-freiburg-template"
DATASET_DIR="/work/dlclarge2/alidemaa-dl_lab/automl/automl-exam-ss25-tabular-freiburg-template/data/superconductivity"
MODEL_DIR="/work/dlclarge2/alidemaa-dl_lab/automl/automl-exam-ss25-tabular-freiburg-template/models/tabpfn-output/superconductivity"
# ---------------------------------------------------------------------------

# sanity checks
if [[ ! -d "$DATASET_DIR" ]]; then
  echo "[ERROR] DATASET_DIR ($DATASET_DIR) does not exist." >&2; exit 1; fi
mkdir -p "$MODEL_DIR" "$MODEL_DIR/preds"

# ── load CUDA (adjust to your cluster) --------------------------------------
module load cuda/11.7

# ── go to project root -------------------------------------------------------
cd "$PROJECT_ROOT" || { echo "[ERROR] Could not cd to $PROJECT_ROOT" >&2; exit 1; }
source automl-tabular-env/bin/activate

# ── sanity check: PyTorch sees the GPU --------------------------------------
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("GPU name :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a")
PY

# ── run TabPFN pipeline ------------------------------------------------------
cd src/models/tabpfn

python tabpfn_predict.py \
    -d "$DATASET_DIR" \
    -m "$MODEL_DIR/ensemble.pkl" \
    -o "$MODEL_DIR/preds/ensemble_preds.csv"

# ── done --------------------------------------------------------------------
# Per‑fold models and ensemble.pkl are now in $MODEL_DIR

#!/bin/bash
#SBATCH --job-name=tabpfn-train
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# ── load CUDA (adjust to your cluster) ─────────────────────────────────
module load cuda/11.7

# ── go to your project root ─────────────────────────────────────────────
cd /work/dlclarge2/alidemaa-dl_lab/automl/automl-exam-ss25-tabular-freiburg-template
source automl-tabular-env/bin/activate
# ── sanity check: PyTorch sees the GPU ─────────────────────────────────
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("GPU name :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a")
PY
cd src/
# ── run TabPFN pipeline ─────────────────────────────────────────────────
# assumes `src/` is on PYTHONPATH
export DATASET_DIR="/work/dlclarge2/alidemaa-dl_lab/automl/automl-exam-ss25-tabular-freiburg-template/data/bike_sharing_demand"
export MODEL_DIR="/work/dlclarge2/alidemaa-dl_lab/automl/automl-exam-ss25-tabular-freiburg-template/models/tabpfn-output"

python -m models.bootstrap_tabpfn \
  --dataset "$DATASET_DIR" \
  --out-dir "$MODEL_DIR" \
  --seed 0 \
  --fold 5 \
  --optuna \
  --n-trials 50

# ── after the job, your per-fold models and ensemble.pkl will be in $MODEL_DIR

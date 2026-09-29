#!/bin/bash
#SBATCH --job-name=prismt_hpo_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Conda env: check/create/activate
ENV_NAME="prismt"
eval "$(conda shell.bash hook)"
if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  echo "Conda env $ENV_NAME exists, activating..."
else
  echo "Creating conda env $ENV_NAME..."
  conda create -n "$ENV_NAME" python=3.10 -y
  conda run -n "$ENV_NAME" pip install -r "$PROJECT_ROOT/requirements.txt"
fi
conda activate "$ENV_NAME"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export CUDA_VISIBLE_DEVICES=0

OUTDIR="${OUTDIR:-$PROJECT_ROOT/hpo_test}"
mkdir -p "$OUTDIR" logs

# Edit DATA_PATH_HERE to point to your standardized .mat file on the cluster
python hpo_optuna.py --data_path "DATA_PATH_HERE" --data_type dff \
  --task_mode classification --target_column phase --phase1 early --phase2 late \
  --n_trials 5 --max_epochs 10 --val_split 0.2 --seed 42 \
  --out_dir "$OUTDIR" --storage "sqlite:///$OUTDIR/study.db"

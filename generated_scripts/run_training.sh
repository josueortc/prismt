#!/bin/bash
#SBATCH --job-name=prismt_phase
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# Project root: resolved from script location (works on cluster)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Optional setup (conda, modules)
# conda activate myenv

python train.py --data_path "/path/to/data.mat" --data_type dff --task_type phase --phase1 early --phase2 late --batch_size 16 --epochs 100 --learning_rate 5e-5 --weight_decay 1e-3 --val_split 0.20 --seed 42 --hidden_dim 128 --num_heads 4 --num_layers 3 --ff_dim 256 --dropout 0.30 --scheduler_type cosine_warmup --warmup_epochs 5 --save_dir results

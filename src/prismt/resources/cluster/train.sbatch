#!/bin/bash
# One training run (or one cross-validation fold when submitted as an array).
source "${SLURM_SUBMIT_DIR:?Submit with: bash submit.sh}/common.sh"
FOLD_ARG=""
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then FOLD_ARG="--fold $((SLURM_ARRAY_TASK_ID + 1))"; fi
exec $PY -m prismt train --config config.json --run-dir results $FOLD_ARG

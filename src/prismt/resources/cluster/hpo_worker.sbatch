#!/bin/bash
# One automatic-tuning worker; many run in parallel and share results/hpo/study.journal.
source "${SLURM_SUBMIT_DIR:?Submit with: bash submit.sh}/common.sh"
exec $PY -m prismt hpo --config config.json --run-dir results --worker "${SLURM_ARRAY_TASK_ID:-0}"

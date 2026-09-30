#!/bin/bash
# Runs after every tuning worker has ended: retrains the best settings and tests them once.
source "${SLURM_SUBMIT_DIR:?Submit with: bash submit.sh}/common.sh"
exec $PY -m prismt hpo --config config.json --run-dir results --finalize

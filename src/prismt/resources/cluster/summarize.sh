#!/bin/bash
# Combine cross-validation folds that ran as separate jobs:   bash summarize.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/job.env"
export SLURM_SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SLURM_SUBMIT_DIR/common.sh"
exec $PY -m prismt train --config config.json --run-dir results --combine

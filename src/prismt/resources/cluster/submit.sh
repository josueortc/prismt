#!/bin/bash
# Submit this PRISMT job:   bash submit.sh
# Safe to run again: finished tuning trials are kept and the study continues.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./job.env
mkdir -p logs results     # Slurm needs logs/ to exist before the job starts
OPTS=(--job-name "prismt-$PRISMT_JOB_NAME" --time "$PRISMT_TIME" --cpus-per-task "$PRISMT_CPUS" --mem "$PRISMT_MEM"
      --chdir "$PWD" --export "ALL" --open-mode append)
[ -n "${PRISMT_PARTITION:-}" ] && OPTS+=(--partition "$PRISMT_PARTITION")
[ -n "${PRISMT_ACCOUNT:-}" ] && OPTS+=(--account "$PRISMT_ACCOUNT")
[ -n "${PRISMT_QOS:-}" ] && OPTS+=(--qos "$PRISMT_QOS")
[ -n "${PRISMT_CONSTRAINT:-}" ] && OPTS+=(--constraint "$PRISMT_CONSTRAINT")
[ "${PRISMT_GPUS:-0}" -gt 0 ] && OPTS+=(--gpus "$PRISMT_GPUS")   # omitted entirely without GPUs
if [ "$PRISMT_MODE" = "hpo" ]; then
  W=$(sbatch --parsable "${OPTS[@]}" --array "0-$((PRISMT_HPO_WORKERS - 1))" --output "logs/hpo_%A_%a.out" hpo_worker.sbatch)
  F=$(sbatch --parsable "${OPTS[@]}" --dependency "afterany:${W%%;*}" --output "logs/final_%j.out" hpo_final.sbatch)
  echo "Submitted tuning workers (job ${W%%;*}) and the final retraining (job ${F%%;*})."
elif [ "${PRISMT_FOLDS:-1}" -gt 1 ]; then
  J=$(sbatch --parsable "${OPTS[@]}" --array "0-$((PRISMT_FOLDS - 1))" --output "logs/fold_%A_%a.out" train.sbatch)
  echo "Submitted $PRISMT_FOLDS cross-validation folds as job ${J%%;*}."
  echo "When they have all finished, combine them with:  bash summarize.sh"
else
  J=$(sbatch --parsable "${OPTS[@]}" --output "logs/train_%j.out" train.sbatch)
  echo "Submitted training job ${J%%;*}."
fi
echo "Check progress with:  squeue --me    and    tail -f logs/*.out"

#!/bin/bash
# Shared set-up for every PRISMT cluster job. Sourced by the .sbatch files as
#   source "${SLURM_SUBMIT_DIR:?}/common.sh"
# Paths come from SLURM_SUBMIT_DIR, never from $0: under sbatch, $0 is Slurm's spooled
# copy of the script, so paths relative to it point to the wrong folder.
set -eo pipefail   # not -u: conda's activation scripts use unset variables
cd "${SLURM_SUBMIT_DIR:?Submit with: bash submit.sh}"
source ./job.env

export PYTHONPATH="$PWD/prismt_src/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

if [ -n "${PRISMT_PYTHON:-}" ]; then
  PY="$PRISMT_PYTHON"
elif [ "${PRISMT_RUNTIME:-conda}" = "apptainer" ]; then
  PY="apptainer exec --nv --bind $PWD $PRISMT_IMAGE python"
else
  if [ -n "${PRISMT_MODULES:-}" ]; then module load $PRISMT_MODULES; fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "PRISMT: conda was not found. Add the module that provides it (e.g. miniconda) to the cluster profile." >&2
    exit 4
  fi
  eval "$(conda shell.bash hook)"
  if ! conda activate "$PRISMT_CONDA_ENV" 2>/dev/null; then
    echo "PRISMT: the environment '$PRISMT_CONDA_ENV' does not exist on this cluster yet." >&2
    echo "PRISMT: run once, on a login node:  bash setup_env.sh" >&2
    exit 4
  fi
  PY=python
fi
# Never touch CUDA_VISIBLE_DEVICES: Slurm sets it to the GPUs this job owns.
$PY -m prismt doctor --require "${PRISMT_DEVICE:-auto}" || { echo "PRISMT: environment check failed (see above)." >&2; exit 4; }

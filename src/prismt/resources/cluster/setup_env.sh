#!/bin/bash
# One-time set-up of the PRISMT Python environment on this cluster. Run on a LOGIN node:
#   bash setup_env.sh
# Installs the CUDA build of PyTorch that matches most cluster GPUs (override with
# PRISMT_TORCH_VERSION / PRISMT_CUDA_TAG in job.env if yours needs another).
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./job.env
if [ "${PRISMT_RUNTIME:-conda}" = "apptainer" ]; then
  apptainer pull "$PRISMT_IMAGE" "$PRISMT_IMAGE_URI"
  exit 0
fi
if [ -n "${PRISMT_MODULES:-}" ]; then module load $PRISMT_MODULES; fi
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$PRISMT_CONDA_ENV"; then
  conda create -y -n "$PRISMT_CONDA_ENV" -c conda-forge --override-channels python=3.11 pip
fi
conda activate "$PRISMT_CONDA_ENV"
pip install "torch==${PRISMT_TORCH_VERSION:-2.5.1}" --index-url "https://download.pytorch.org/whl/${PRISMT_CUDA_TAG:-cu121}"
pip install numpy scipy h5py scikit-learn optuna
PYTHONPATH="$PWD/prismt_src/src" python -m prismt doctor || true   # no GPU on a login node is expected
echo "PRISMT environment '$PRISMT_CONDA_ENV' is ready. Now run:  bash submit.sh"

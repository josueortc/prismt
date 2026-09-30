#!/bin/bash
# Run a PRISMT job folder under the fake Slurm:  bash tests/fake_slurm/run_jobfolder.sh <job folder>
# Needs PRISMT_PYTHON (a Python with PRISMT's dependencies).
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HERE:$PATH"
: "${PRISMT_PYTHON:?set PRISMT_PYTHON}"
export PRISMT_PYTHON
bash "$1/submit.sh"

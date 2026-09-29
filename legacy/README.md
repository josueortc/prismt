# Legacy code (pre-rebuild)

Everything in this folder is the PRISMT code as it was before the rebuild on branch
`rebuild/v1`. It is kept for reference only: nothing in the new package imports it, it is
not on the MATLAB path, and it will be deleted in v0.2.

The committed Python in this folder cannot train. Several problems also silently changed
results, so please do not use it for analysis.

## Where things went

| Old | New |
|---|---|
| `run_prismt_gui.m` → `gui/prismt_training_setup.m` | `run_prismt_gui.m` → `prismt.gui()` (tabbed app in `matlab/+prismt/`) |
| `scripts/standardize_data.m`, `utils/preprocess_*.m` | Data tab → Import…, or `prismt.importData(file, ...)`, which writes one PRISMT dataset file |
| `gui/validate_prismt_mat.m`, `scripts/validate_data.py` | `python -m prismt validate <dataset.mat>` (also run by the app) |
| `python train.py --data_path … --phase1 early --phase2 late …` | `python -m prismt train --config run/config.json` (the app writes the config) |
| `python hpo_optuna.py …` | `python -m prismt hpo --config …` ("Tune automatically" in the app) |
| `python analyze_results.py --results_dir …` | Results tab, or `prismt.loadResults(runFolder)` + `prismt.plot.*` |
| `scripts/run_optuna_cluster.sh`, generated `run_training.sh` | Run tab → Cluster → Create job folder (see `docs/cluster.md`) |
| `requirements.txt` | `environment.yml` and `pyproject.toml` |
| `docs/wiki/Hyperparameter-Guide.md` | `docs/hyperparameters.md` |

## Known problems in this code

Data loading (`data/data_loader.py`, `train.py`):
- Importing fails with `NameError: name 'Any' is not defined` (data_loader.py:1753).
- `create_data_loaders_unified` uses arguments it does not receive (train.py:101-160), and
  the first batch is unpacked as 2 values although batches have 3 (train.py:767,
  hpo_optuna.py:418).
- `processed_data` saved as v7 loads zero sessions; the table-`T` fallback invents phase
  labels; array orientation is guessed from sizes, which is wrong for v7.3 files with more
  than 50 trials per session.
- Classifying by `stim` or `response` actually learns phase; the `mouse` filter is ignored;
  choosing two mice labels every mouse by its `wt_`/`mut_` prefix.
- There is no test split: the validation mice choose the checkpoint, drive early stopping
  and HPO, and produce the final report.

Model and training (`models/`, `training/`):
- The "causal" attention multiplies a 0/1 mask after the softmax, so future time bins
  still influence each token and attention rows no longer sum to 1.
- The masked-autoencoder model has no training loop, loss or evaluation.
- `torch.load` without `weights_only=False` fails on torch 2.6 and newer;
  `ReduceLROnPlateau(verbose=True)` fails on torch 2.11 and newer; wandb is always started,
  with a personal default entity.

MATLAB (`gui/`, `scripts/`, `utils/`):
- The CDKL5 scripts cut continuous recordings into trials with
  `reshape(X, [n_trials, 30, R])` on a time × regions matrix, so each "trial" interleaves
  samples from across the whole recording (standardize_data.m:241,357;
  preprocess_cdkl5_data.m:260,275,576,591).
- `standardize_data.m` saves `standardized_data`, which the GUI and the loader reject.
- The GUI's Normalization dropdown is never used; `Run Training Now` blocks MATLAB for the
  whole run; the fallback layout crashes (`get(0,'ScreenSize',3)`).
- Generated SLURM scripts find the project from `$0` (wrong under `sbatch`), need `logs/`
  to exist before submission, build the conda environment inside the GPU job, and force
  `CUDA_VISIBLE_DEVICES=0`.

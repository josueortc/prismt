# Contracts between MATLAB and Python

MATLAB and Python never call each other's code. They exchange four kinds of files and
commands, each versioned (`src/prismt/resources/formats.json`), so either side can change
internally as long as these stay the same. A change to any of them bumps its version and
needs matching changes and tests on both sides.

| # | Contract | Written by | Read by | Version key | Tests |
|---|---|---|---|---|---|
| 1 | Dataset file | MATLAB (`prismt.writeDataset`), Python (`write_dataset`, `.h5`/`.npz`) | Python (`read_dataset`), MATLAB (`prismt.loadDataset`) | `dataset` | `tests/fixtures/dataset_v1_matlab*.mat` (written by MATLAB, read by `test_io.py`); `tCrossLanguage.m` |
| 2 | Run settings (`run.json` → `config.json`) | MATLAB (partial) | Python (`load_config`) | `config` | `test_pipeline.py` (config tests), `tBackend.m`, `tAppController.m` |
| 3 | Run folder | Python | MATLAB (`prismt.loadResults`, `LocalRun`, plots) | `run`, `status`, `results` | `test_pipeline.py`, `tBackend.m`, `tAppSmoke.m` |
| 4 | Command line (`python -m prismt ...`) | — | MATLAB (`prismt.run.runPython`, `LocalRun`), SLURM templates | CLI version = package version | `test_cli.py`, `test_cluster.py`, `tBackend.m` |

## 1. Dataset file

Specified for users in [../data-format.md](../data-format.md). For implementers:

- h5py sees MATLAB arrays with the dimensions reversed. `io/matv73.py` transposes them
  and restores trailing singleton dimensions from `meta.shape`, and
  `meta.orientation_probes` (values of `X` at a few 0-based indices, computed by MATLAB)
  are checked after the transpose, so an orientation mistake fails loudly instead of
  silently permuting channels and time.
- `channels.groups` (optional): one text per channel ("" for none); `modalities.kinds`: any
  non-empty text (usual values: neural, behavior, physiology, stimulus, signal, other).
- Text is stored as `uint8` UTF-8, never as MATLAB `char` (UTF-16 in HDF5 and awkward in
  h5py).
- `Inf` is an error (reported with the MATLAB index); `NaN` means missing.
- Unknown or future `prismt_version` values are refused with an explanation.

## 2. Run settings

- Every setting is declared once, in `src/prismt/resources/config_schema.json`: path,
  type, default, limits, label, help text, `advanced` flag. Python validates against it,
  the app builds its controls and tooltips from it, and `docs/hyperparameters.md` is
  generated from it (`tools/make_settings_doc.py`; a test checks the page is current).
- Resolution order: schema defaults < preset < user values. Unknown keys are errors with a
  "did you mean" suggestion.
- MATLAB's `jsonencode` turns 1-element lists into scalars and empty lists into `[]`;
  Python accepts a scalar wherever a list is expected. `NaN` is written as `null`.
- Relative paths are resolved against the folder of the config file. Job folders rely on
  this: `dataset.path = "data/x.mat"`, `output.root = "results"`.
- Fine-tuning is `task = "classify"` with `model.init_from = <autoencoder run folder>`.
- The run folder keeps both `run.json` (as given) and `config.json` (resolved, plus the
  config hash and the dataset and split fingerprints).

## 3. Run folder

`<root>/<task>-<YYYYmmdd-HHMMSS>-<random>[-name]/`. Python never writes into a folder that
already holds results (`output.overwrite` aside); MATLAB may create the folder and
`run.json` first.

| File | Written | Content |
|---|---|---|
| `status.json` | throughout, atomically | `schema = "prismt.status/1"`; `state` ∈ queued, preparing, training, evaluating, finished, failed, cancelled; `message`; `fold`, `n_folds`, `epoch`, `epochs`, `step`, `best_epoch`, `best_value`, `latest`, `eta_s`, `elapsed_s`, `pid`, `host`; `error = {code, title, message, hint, field}` when failed |
| `history.csv` | each epoch | `fold, epoch, lr, train_loss, val_loss, val_<metric>..., seconds, is_best` (booleans as 1/0) |
| `log.txt`, `stdout.txt` | throughout | Python's log, and everything printed |
| `config.json`, `run.json`, `run_info.json` | at start | resolved settings, given settings, versions / device / git commit |
| `splits.csv` | at start | `trial_index` (1-based, dataset order), `split`, `fold`, `subject`, `session`, `label` |
| `metrics.json` | at end | `task`, `headline {name, value, chance, description}`, `test`, `val`, `folds`, `baselines`, `by_group`, `summary_lines` (plain sentences shown by the app), `warnings`, `flags` |
| `predictions.csv` | at end (classification) | per test trial: index, fold, true and predicted class, probability of each class |
| `results.mat` | at end | scipy v5 `.mat` for `load`: class names, predictions, confusion; for the autoencoder, R² per mask / channel / time, baseline R², per-trial sums (`n`, `s1`, `s2`, `sse`) for R² by any grouping, reconstruction examples; embeddings; channel and time information |
| `model.pt` | at end | loads with `torch.load(weights_only=True)`: state dict, settings as JSON text, normalization, class names, data signature |
| `fold_XX/` | cross-validation | the same files for one fold (and the only files, when folds run as separate cluster jobs until `train --combine`) |
| `hpo/`, `final/`, `hpo_summary.json`, `trials.csv`, `best_config.json` | tuning runs | Optuna journal, retrained seeds, summary |
| `STOP` | by MATLAB | `stop` (default): stop after the current step and evaluate the best model so far; `cancel`: stop without results |

A process that dies without writing a final state is detected by MATLAB
(`LocalRun.status`: the process is gone and the state is not final) and explained from the
log (`+run/explainFailure.m`, `+run/known_errors.json`).

## 4. Command line

`python -m prismt <command>`; with `--json`, stdout carries exactly one JSON object and
logs go to stderr.

| Command | Purpose |
|---|---|
| `doctor [--json] [--require DEVICE]` | Environment report (versions, device); non-zero exit when unusable |
| `validate FILE [--json]` | Check a dataset file |
| `synth --profile P --out FILE` | Demo data (same recipe as `prismt.demo.makeSyntheticDataset`) |
| `defaults --json` | The schema, presets and resolved defaults |
| `check --config C [--timing] [--json]` | What a run would do: selection, class counts, split plan, model size, time estimate, warnings |
| `train --config C [--run-dir D] [--fold K] [--combine]` | Train; `--fold` for one cluster array task, `--combine` to pool finished folds |
| `hpo --config C [--run-dir D] [--worker I \| --finalize]` | Automatic tuning; workers share the study in `D/hpo/` |
| `jobfolder --config C --profile P --out DIR [--mode train\|hpo] [--workers N] [--remote-dataset PATH]` | Write a SLURM job folder |
| `summarize RUN` | Print a run's summary |

Exit codes: 0 ok, 1 internal error, 2 problem with the data or settings (the message says
what to change), 3 cancelled, 4 environment problem. Signals: SIGUSR1 or a `STOP` file →
stop gracefully and evaluate; SIGTERM or `STOP` containing `cancel` → stop without results.

MATLAB launches Python with `java.lang.ProcessBuilder` and an argument array (no shell
quoting), a cleaned environment (MATLAB's library paths and variables removed) and
`PYTHONPATH=<repo>/src`, so the Python code always matches the app. Without a JVM it falls
back to the generated `launch.sh` / `launch.cmd`.

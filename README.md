# PRISMT

PRISMT trains transformer models on trial-based recordings from MATLAB, without needing to
know about GPUs, clusters, hyperparameter tuning or transformers. A recording can be any set
of signals over time: brain activity (imaging, electrodes), behavior (running speed, pupil,
body-part positions), physiology, or stimulus traces, alone or together, with any number of
channels. It can:

- **classify trial conditions** (e.g. CS+ vs CS−, early vs late learning, hit vs miss) and
  test the classifier on subjects (animals, participants) it never saw, with
  cross-validation so every subject is tested once;
- **learn the structure of the signals** with a masked autoencoder: hide part of each trial,
  predict it from the rest, and see which channels, times and signals are predictable;
- **combine the two**: start a classifier from a trained autoencoder when labelled trials are few.

Every result is shown next to simple reference models (chance, the most common class,
logistic regression; for the autoencoder, trial averages and neighbouring values), so it is
clear when the transformer adds something.

The model is the backbone of the PRISMt paper (masked autoencoding with block-causal
attention over channel × time tokens). The paper's interpretation tools (attribution, CP
motifs) are not included yet.

## Install

You need MATLAB R2021a or newer (no toolboxes) and about 3 GB of disk space.

1. Get the code: `git clone https://github.com/josueortc/prismt.git` (or download the ZIP).
2. In MATLAB, open the `prismt` folder and double-click **`run_prismt_gui.m`**.
3. On the **Setup** tab, press **Create environment**. PRISMT installs its own Python with
   PyTorch (5–15 minutes, once per computer; it uses conda if you have it, otherwise it
   downloads micromamba, and needs no administrator rights). If you already have a
   suitable environment, **Find automatically** picks it up.

## First run (5 minutes)

1. **Data** tab > **Create demo data**. This makes a synthetic experiment with a planted
   stimulus response and a planted learning effect, so you know what the answer should be.
2. **Task** tab > **Classify trial conditions**, *Predict* `phase` (early vs late), then
   **Check settings**. The Checks list says how trials will be split for testing.
3. **Run** tab > **Start training**. MATLAB stays usable; the learning curves update live.
4. **Results** tab: the score against the reference models, the confusion matrix, accuracy
   per subject, and more.

Then try **Learn structure (masked autoencoder)** on the same data, and open
*Predictability per channel* in the results.

The full walk-through, including your own data, is in [docs/tutorial.md](docs/tutorial.md).

## Your own data

The **Data** tab imports the lab's `tableForModeling` tables, `processed_data` /
`standardized_data` structs, numbered-variable files and CDKL5 recordings (with options for
which signals to use, how channels are split into signals, and behavior columns), or any
arrays with `prismt.makeDataset`:

```matlab
addpath('matlab')
% X: trials x channels x time (x signals); trials: a table with one row per trial
ds = prismt.makeDataset(X, trials, SamplingRate=10, TimeZero=-1, ...
                        Subject="mouse", Session="session", ChannelNames=names);
prismt.writeDataset(ds, "mydata_prismt.mat");
```

Each signal can have its own channels (e.g. 64 electrodes and 3 behavior variables), channels
can be given groups (areas, sides, sensors) to train on some of them, and recordings whose
channels differ are joined with **Add dataset...** or `prismt.combineDatasets` (channels are
matched by name; missing ones are skipped by the model). What a dataset file contains is
described in [docs/data-format.md](docs/data-format.md).

## Scripts instead of the app

Everything the app does is a MATLAB function, and **Export as MATLAB script** (Run or
Results tab) writes the script for any run:

```matlab
addpath('matlab')
cfg = prismt.defaultConfig("classify", "mydata_prismt.mat", Label="phase");
report = prismt.check(cfg);             % classes, test split, model size; problems explained
run = prismt.train(cfg, Wait=true);     % runs in the background; Wait prints progress
R = prismt.loadResults(run.RunDir);
figure; prismt.plot.scoreVsBaselines(gca, R);
```

[matlab/examples/prismt_tutorial.m](matlab/examples/prismt_tutorial.m) runs all three
tasks on demo data and draws the figures. Python users can use the same engine directly:
`python -m prismt --help`.

## Where to go next

| Document | What it covers |
|---|---|
| [docs/tutorial.md](docs/tutorial.md) | The app, step by step |
| [docs/results-and-pitfalls.md](docs/results-and-pitfalls.md) | Reading the results, and the mistakes PRISMT guards against |
| [docs/hyperparameters.md](docs/hyperparameters.md) | Every setting, its default, and what to change first |
| [docs/cluster.md](docs/cluster.md) | Training on a SLURM cluster (job folders, copy-paste commands) |
| [docs/containers.md](docs/containers.md) | Optional Docker / Apptainer image |
| [docs/data-format.md](docs/data-format.md) | The dataset file |
| [docs/dev/contracts.md](docs/dev/contracts.md) | For developers: the files MATLAB and Python exchange |
| [docs/dev/testing.md](docs/dev/testing.md) | For developers: running the tests |
| [legacy/README.md](legacy/README.md) | The previous version of this repository, and what replaced each part |

## Layout of this repository

```text
run_prismt_gui.m        opens the app
matlab/+prismt/         MATLAB package: app, import, datasets, runs, plots
matlab/examples/        tutorial script
src/prismt/             Python engine: data, model, training, tuning, cluster templates
tests/                  MATLAB and Python tests, shared fixtures, a fake SLURM for tests
docker/Dockerfile       optional container image
docs/                   documentation
legacy/                 the previous version (to be removed in 0.2)
```

## Status

Version 0.1 (in development). Tested on macOS with MATLAB R2025a, Python 3.11 and
PyTorch 2.5.1 (Apple GPU and CPU); the cluster scripts are tested against a simulated
SLURM. Windows and Linux are expected to work but are not yet tested routinely.

## License

To be decided by the authors.

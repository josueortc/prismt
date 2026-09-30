# Running the tests

Tests never use real data: they build synthetic datasets with known structure
(`prismt.demo.makeSyntheticDataset` / `python -m prismt synth`), so a test can check that
a model finds what was planted.

## Python

```bash
conda env create -f environment.yml        # once; or use the app's Create environment
conda activate prismt
python -m pytest tests/python -q -m "not slow"   # about 5 s
python -m pytest tests/python -q -m slow         # about 1 min: whole runs, tuning, fake SLURM
python tools/make_settings_doc.py --check        # docs/hyperparameters.md matches the schema
```

| File | Covers |
|---|---|
| `test_io.py` | dataset files: MATLAB-written fixtures, orientation probes, validation messages |
| `test_synthetic.py` | the demo recipe matches `tests/fixtures/synthetic_structure_v1.json`; simple estimators find the planted effects |
| `test_model.py` | attention masks (no future leakage, CLS read-only, NaN safety), masking, checkpoints |
| `test_pipeline.py` | settings, selection, splits (no animal in two parts), whole runs, fine-tuning, shuffled-label baseline |
| `test_cli.py` | the command line as MATLAB uses it; SIGUSR1 stops a run with results |
| `test_cluster.py` | job folders run end to end under a fake SLURM (`tests/fake_slurm/sbatch`) |

## MATLAB

From the repository folder (with `PRISMT_PYTHON` set to the Python of a PRISMT
environment, for the tests tagged `Python`):

```bash
PRISMT_PYTHON=/path/to/envs/prismt/bin/python \
  matlab -batch "addpath('tests/matlab'); r = runPrismtTests(); exit(double(any([r.Failed])))" > test.log 2>&1
```

Redirect the output to a file as shown; piping `matlab -batch` into another program can
hang on macOS. In MATLAB itself: `addpath('tests/matlab'); runPrismtTests()`.

| Test class | Tags | Covers |
|---|---|---|
| `tDataset`, `tSynthetic` | | datasets, validation, the demo recipe |
| `tCrossLanguage` | Python | MATLAB writes, Python reads and checks, and back |
| `tBackend` | Python | environment check, background runs, stop, failures explained, importers, job folders |
| `tAppController` | | the app's logic without a window: classes, filters, checks, script export |
| `tAppSmoke` | UI, Python, Slow | the hidden app driven through every task, with screenshots |
| `tCompatibility` | | no functions newer than R2021a; no toolboxes needed |
| `tRealData` | Python, RealData | opt-in: set `PRISMT_REAL_DATA` to a `tableForModeling_v2`-style file; imports it and checks a classification (real data is never committed) |

`runPrismtTests(Exclude="Python")` skips the tests that need Python;
`runPrismtTests(Name="tAppSmoke")` runs one class. Set `PRISMT_TEST_PNG_DIR` to keep the
app screenshots `tAppSmoke` takes (useful after changing a tab's layout: look at them).

After changing the MATLAB dataset writer, regenerate the fixtures Python reads:
`addpath('tests/matlab'); makeFixtures()`.

## Continuous integration

`.github/workflows/` runs the Python tests (pinned and latest dependencies), the MATLAB
tests on R2021a and the latest release, and builds the container images.

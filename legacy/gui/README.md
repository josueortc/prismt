# PRISMT Training Setup GUI

Suite2P-style MATLAB GUI for configuring and launching PRISMT training on widefield calcium imaging data.

## Quick Start (easiest)

**From project root**, double-click:
```
run_prismt_gui.m
```
Or in MATLAB Command Window:
```matlab
run_prismt_gui
```

No need to `cd` into any folder.

## Alternative

```matlab
cd gui
prismt_training_setup()
```

## Requirements

- MATLAB R2016a or newer (uses `uifigure` for modern UI)
- Standardized `.mat` file (from `scripts/standardize_data.m`)

## Workflow

1. **Load Dataset** – Browse to your standardized `.mat` file and click Validate
2. **Input & Normalization** – Choose dff/zscore and normalization (scale_20, robust_scaler, etc.)
3. **Tokenization** – Spatial scheme (brain areas as tokens) is the default
4. **Conditions** – Select phase comparison (early vs late) or genotype task, plus stim/response filters
5. **Generate Script** – Set batch size, epochs, cluster options; generate SLURM script

## Generated Outputs

- `run_training.sh` – SLURM submission script
- `train_config.txt` – Human-readable configuration summary
- Copy-paste ready Python command for local execution

## Widefield Dataset Example

For widefield data with phases (early, mid, late):

1. Load your standardized file (e.g. from `standardize_data('widefield_raw.mat', 'standardized.mat', 'widefield')`)
2. Select **Phase (early vs late)** as task type
3. Choose phases, e.g. **early** vs **late**
4. Use stim=1, response=0,1 as typical filters
5. Generate script and submit: `sbatch generated_scripts/run_training.sh`

## Design Document

See [../docs/PRISMT_GUI_DESIGN.md](../docs/PRISMT_GUI_DESIGN.md) for full design specification.

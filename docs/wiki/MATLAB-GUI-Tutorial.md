# MATLAB GUI Tutorial

**Frame-by-frame guide to using the PRISMT Training Setup GUI.**

---

## Quick Navigation

- [Launching the GUI](#launching-the-gui)
- [Step 1: Load Dataset](#step-1-load-dataset)
- [Step 2: Input & Tokenization](#step-2-input--tokenization)
- [Step 3: Comparison Conditions](#step-3-comparison-conditions)
- [Step 4–6: Training, Model, Cluster](#step-4-6-training-model-cluster)
- [Step 7: Run](#step-7-run)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Generating Screenshots](#generating-screenshots)

---

## Launching the GUI

From the project root in MATLAB:

```matlab
run_prismt_gui
```

Or double-click `run_prismt_gui.m` in the Current Folder.

![Initial GUI](images/gui/1_initial.png)

*Figure 1: Initial GUI state*

---

## Step 1: Load Dataset

### Panel 1: Load Dataset

| Control | Action |
|---------|--------|
| **Dataset (.mat)** | Path to your standardized `.mat` file |
| **Browse** | Opens file dialog to select a `.mat` file |
| **Load** | Validates the file and loads metadata |

### 1a. Click Browse

1. Click **Browse**
2. Navigate to your standardized `.mat` file
3. Select it and click Open

The path appears in the Dataset field.

![After Browse](images/gui/2_path_entered.png)

### 1b. Click Load

1. Click **Load**
2. The GUI validates the file structure
3. On success: summary shows `OK: N datasets | ~M trials x T timepoints x R regions`
4. Condition dropdowns (Phases, Stim, Response) are populated from your data

![After Load](images/gui/3_loaded.png)

*Figure 3: Data loaded and validated*

**Creating sample data for testing:**

```matlab
cd scripts
outPath = create_sample_for_docs();
% Then in GUI: Browse -> select outPath -> Load
```

---

## Step 2: Input & Tokenization

**Panel 2** configures how neural data is used.

| Setting | Options | Default |
|---------|---------|---------|
| Data type | dff (ΔF/F), zscore | dff |
| Normalization | Scale ×20, Robust, Percentile clip, None | Scale ×20 |
| Tokenization | Spatial (1 token per region) | Computed after load |

After loading, Tokenization shows: `Spatial: N regions -> N tokens (+ CLS)`.

---

## Step 3: Comparison Conditions

**Panel 3** defines what you are comparing.

### Task type

- **Phase (early vs late)** – Compare trial phases
- **Genotype (WT vs mutant)** – Compare genotypes (CDKL5)

### For Phase classification

| Field | Description |
|-------|-------------|
| Phases | Phase 1 vs Phase 2 (e.g., early vs late) |
| Stim | Comma-separated stimulus values (e.g., `1`) |
| Response | Comma-separated response values (e.g., `0, 1`) |
| Seed | Random seed for train/val split |

### For Genotype classification

Phases are ignored. Mouse IDs define the classes.

![Conditions](images/gui/4_conditions.png)

*Figure 4: Conditions configured*

---

## Step 4–6: Training, Model, Cluster

### Panel 4: Training

| Parameter | Default |
|-----------|---------|
| Batch | 16 |
| Epochs | 100 |
| LR | 5e-5 |
| Weight decay | 1e-3 |
| Val split | 0.2 |
| Save dir | results |

### Panel 5: Model

| Parameter | Default |
|-----------|---------|
| Hidden | 128 |
| Heads | 4 |
| Layers | 3 |
| FF dim | 256 |
| Dropout | 0.3 |
| Scheduler | cosine_warmup |
| Warmup | 5 |

### Panel 6: Cluster (SLURM)

| Field | Default |
|-------|---------|
| Partition | gpu |
| GPUs | 1 |
| CPUs | 8 |
| Mem | 32G |
| Time(hr) | 24 |
| Data path on cluster | (empty) |
| HPO out dir (cluster) | (empty) |
| Setup | (empty) |

---

## Step 7: Run

**Action panel (right side):**

| Button | Action |
|--------|--------|
| **Run Training Now** | Runs training locally |
| **Generate Cluster Script** | Creates `run_training.sh` and `train_config.txt` |

![Ready](images/gui/5_ready.png)

*Figure 5: Ready to run*

---

## Hyperparameter Optimization

When you select **HPO (Optuna)** in the Mode dropdown, PRISMT runs an automatic search over learning rate, model size, dropout, and other hyperparameters. Each trial trains a different configuration; the best one is saved.

For a didactic explanation of each hyperparameter—what it controls and how changing it helps—see the [Hyperparameter Guide](Hyperparameter-Guide).

For a minimal cluster script to test Optuna HPO (conda env setup, single GPU, quick trials), see [scripts/run_optuna_cluster.sh](../../scripts/run_optuna_cluster.sh) in the main repo.

---

## Generating Screenshots

To capture screenshots for documentation:

```matlab
run_prismt_gui
addpath('scripts')

capture_gui_screenshots('1_initial')      % After launch
% Browse and select .mat
capture_gui_screenshots('2_path_entered')
% Click Load
capture_gui_screenshots('3_loaded')
% Adjust conditions
capture_gui_screenshots('4_conditions')
capture_gui_screenshots('5_ready')
```

Screenshots save to `docs/images/gui/`.

---

## Workflow Summary

```
1. Load Dataset    → Browse → Select .mat → Load
2. Input & Token   → dff/zscore, normalization
3. Conditions      → Task, phases, stim, response
4. Mode           → Standard or HPO
5. Training       → Batch, epochs, LR
6. Model          → Hidden, heads, layers
7. Cluster        → Partition, GPUs, time
8. Run            → Run Now or Generate Script
```

---

## See Also

- [Hyperparameter Guide](Hyperparameter-Guide) – What each hyperparameter does and how to tune it
- [PRISMT GUI Design](PRISMT_GUI_DESIGN.md)
- [Data Standardization Overview](Data-Standardization-Overview)
- [Validation and Troubleshooting](Validation-and-Troubleshooting)

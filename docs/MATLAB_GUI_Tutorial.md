# PRISMT MATLAB GUI – Step-by-Step Tutorial

This guide walks you through the PRISMT Training Setup GUI with frame-by-frame instructions and screenshots.

---

## Launching the GUI

**From project root:**

```matlab
run_prismt_gui
```

Or double-click `run_prismt_gui.m` in the Current Folder.

![Launch](images/gui/1_initial.png)

*Figure 1: Initial GUI state after launch*

---

## Step 1: Load Dataset

### 1.1 Initial state

The GUI opens with six panels. **Panel 1: Load Dataset** is at the top.

| Element | Description |
|---------|-------------|
| Dataset (.mat) | Text field for the path to your standardized `.mat` file |
| **Browse** | Opens file dialog to select a `.mat` file |
| **Load** | Validates the file and loads metadata |
| Summary | Shows "No data loaded" until you Load |

### 1.2 Browse for a file

1. Click **Browse**
2. Navigate to your standardized `.mat` file
3. Select it and click Open

The path appears in the Dataset field. The status bar may show "Path OK. Click Load to validate."

![After Browse](images/gui/2_path_entered.png)

*Figure 2: Path entered after Browse*

### 1.3 Load and validate

1. Click **Load**
2. The GUI validates the file structure
3. On success: summary shows `OK: N datasets | ~M trials x T timepoints x R regions`
4. Condition dropdowns (Phases, Stim, Response) are populated from your data

![After Load](images/gui/3_loaded.png)

*Figure 3: Data loaded and validated*

**Creating sample data for screenshots:**

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

After loading, the Tokenization line shows: `Spatial: N regions -> N tokens (+ CLS)`.

---

## Step 3: Comparison Conditions

**Panel 3** defines what you are comparing.

### Task type

- **Phase (early vs late)** – Compare trial phases (e.g., early vs late learning)
- **Genotype (WT vs mutant)** – Compare genotypes (e.g., CDKL5)

### For Phase classification

| Field | Description |
|-------|-------------|
| Phases | Phase 1 vs Phase 2 – populated from your data (early, mid, late) |
| Stim | Comma-separated stimulus values to include (e.g., `1`) |
| Response | Comma-separated response values (e.g., `0, 1`) |
| Seed | Random seed for train/val split |

### For Genotype classification

Phases are ignored. Mouse IDs (e.g., `wt_*`, `mut_*`) define the classes.

![Conditions selected](images/gui/4_conditions.png)

*Figure 4: Conditions configured (Phase: early vs late)*

---

## Step 4: Training Mode

Choose **Standard training** or **HPO (Optuna)**.

| Mode | HPO trials | Epochs/trial |
|------|------------|--------------|
| Standard | — | — |
| HPO (Optuna) | Number of Optuna trials | Epochs per trial |

---

## Step 5: Training Parameters

**Panel 4: Training**

| Parameter | Default | Description |
|-----------|---------|-------------|
| Batch | 16 | Batch size |
| Epochs | 100 | Training epochs |
| LR | 5e-5 | Learning rate |
| Weight decay | 1e-3 | L2 regularization |
| Val split | 0.2 | Validation fraction |
| Save dir | results | Output directory |

**Panel 5: Model**

| Parameter | Default |
|-----------|---------|
| Hidden | 128 |
| Heads | 4 |
| Layers | 3 |
| FF dim | 256 |
| Dropout | 0.3 |
| Scheduler | cosine_warmup |
| Warmup | 5 |

---

## Step 6: Cluster (SLURM)

**Panel 6** configures cluster submission.

| Field | Default | Description |
|-------|---------|-------------|
| Partition | gpu | SLURM partition |
| GPUs | 1 | GPUs per job |
| CPUs | 8 | CPUs per task |
| Mem | 32 | Memory (GB) |
| Time(hr) | 24 | Wall time |
| Data path on cluster | (empty) | Path to data on cluster |
| HPO out dir (cluster) | (empty) | For HPO: e.g. `$SCRATCH/prismt/optuna_runs/exp1` |
| Setup | (empty) | e.g. `conda activate myenv` |

---

## Step 7: Run

**Action panel (right side):**

| Button | Action |
|--------|--------|
| **Run Training Now** | Runs training locally in MATLAB |
| **Generate Cluster Script** | Creates `run_training.sh` and `train_config.txt` |

![Ready to run](images/gui/5_ready.png)

*Figure 5: Configuration complete, ready to Run or Generate Script*

---

## Generating Screenshots

To capture screenshots for this tutorial:

```matlab
% 1. Launch GUI
run_prismt_gui

% 2. Capture initial state
capture_gui_screenshots('1_initial')

% 3. Browse and select a .mat file, then:
capture_gui_screenshots('2_path_entered')

% 4. Click Load, then:
capture_gui_screenshots('3_loaded')

% 5. Adjust conditions, then:
capture_gui_screenshots('4_conditions')

% 6. Final state:
capture_gui_screenshots('5_ready')
```

Or create sample data first:

```matlab
cd scripts
create_sample_for_docs();
run_prismt_gui
% Browse -> select docs/images/gui/sample_for_docs.mat -> Load
capture_gui_screenshots('3_loaded')
```

Screenshots are saved to `docs/images/gui/`.

---

## Quick Reference: Workflow

```
1. Load Dataset    → Browse → Select .mat → Load
2. Input & Token   → Choose dff/zscore, normalization
3. Conditions      → Task (Phase/Genotype), phases, stim, response
4. Mode           → Standard or HPO
5. Training       → Batch, epochs, LR, save dir
6. Model          → Hidden, heads, layers, etc.
7. Cluster        → Partition, GPUs, time, setup
8. Run            → Run Training Now OR Generate Cluster Script
```

---

## See Also

- [PRISMT GUI Design](PRISMT_GUI_DESIGN.md) – Design specification
- [README](../README.md) – Project overview
- [GitHub Wiki](https://github.com/josueortc/prismt/wiki) – Data standardization guide

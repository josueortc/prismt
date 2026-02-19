# PRISMT Training Setup GUI – Design Document

## Overview

The PRISMT GUI is a MATLAB application designed for **experimental neuroscientists** who need to configure and launch transformer training on widefield calcium imaging data without deep machine learning expertise. The GUI provides an intuitive, step-by-step setup that generates cluster-ready training scripts.

---

## User Persona

- **Primary**: Experimental neuroscientist running calcium imaging experiments
- **Background**: Familiar with MATLAB, trial-based analysis, condition variables (stim, response, phase)
- **ML knowledge**: Limited; needs sensible defaults and clear terminology
- **Goal**: Train a classifier to compare conditions (e.g., early vs. late phase) and run it on a cluster

---

## GUI Layout (Tab-Based Workflow)

### Tab 1: Load Dataset
- **Purpose**: Specify and validate the data file
- **Controls**:
  - **Browse** button → file dialog for `.mat` file
  - **Path** text field (editable)
  - **Validate** button → quick check (number of datasets, shape)
  - **Status** label: "Ready" / "Error: …"
- **Feedback**: After load, show summary (e.g., "42 datasets, ~225 trials each, 41 timepoints × 82 areas")

---

### Tab 2: Define Input
- **Purpose**: Choose what neural signal to use and how to normalize it
- **Controls**:
  - **Data type** dropdown: `dff` | `zscore`
    - Tooltip: "ΔF/F (raw) vs. z-scored activity"
  - **Normalization** dropdown:
    - `Scale by 20` (default) – simple scaling, recommended for dff
    - `Robust (median/IQR)` – robust to outliers, for dff
    - `Percentile clip (5th–95th)` – clip extremes, for zscore
    - `Per-area max` – legacy
    - `None` – raw values
  - **Preview** (optional): Small plot of normalized sample trial

---

### Tab 3: Tokenization Scheme
- **Purpose**: How spatial and time dimensions become tokens (plain-language descriptions)
- **Controls**:
  - **Scheme** dropdown:
    - **Spatial (brain areas as tokens)** [default]
      - Description: "One token per brain area. Time is compressed inside each token."
      - Matches `WidefieldTransformer`
    - *(Future)* **Scalar (time×area tokens)**: "Each timepoint×area pair is a token."
  - **Read-only info**:
    - "Sequence length: N brain areas + 1 (CLS)"
    - "Computed from your data after load"

---

### Tab 4: Conditions & Subset
- **Purpose**: Define comparison conditions and optional subsetting
- **Controls**:
  - **Task type** dropdown: `Phase classification (widefield)` | `Genotype classification (CDKL5)`
  - **For Phase**:
    - Phase 1 dropdown: early | mid | late (from data)
    - Phase 2 dropdown: early | mid | late
  - **For Genotype**: (auto-detected from mouse IDs)
  - **Condition filters**:
    - Stim values: multi-select or comma list (e.g., `1`)
    - Response values: multi-select or comma list (e.g., `0, 1`)
  - **Subset (optional)**:
    - Max datasets to use (e.g., 10 for quick tests)
    - Max trials per dataset (e.g., 100)
    - Random seed for reproducibility

---

### Tab 5: Training & Cluster
- **Purpose**: Set training parameters and generate the cluster script
- **Controls**:
  - **Batch size** (default: 16)
  - **Epochs** (default: 100)
  - **Learning rate** (default: 5e-5)
  - **Validation split** (default: 0.2)
  - **Cluster options**:
    - SLURM partition
    - GPUs per job
    - Memory
    - Job name
  - **Output**:
    - Directory for scripts and config
    - **Generate Script** button
- **Outputs**:
  - `run_training.sh` – SLURM submit script
  - `train_config.txt` – human-readable config summary
  - Command-line string for `train.py` (copy-paste ready)

---

## Generated Artifacts

### 1. `run_training.sh` (SLURM)
```bash
#!/bin/bash
#SBATCH --job-name=prismt_phase
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

python train.py --data_path /path/to/data.mat \
  --data_type dff --task_type phase \
  --phase1 early --phase2 late \
  --batch_size 16 --epochs 100 ...
```

### 2. `train_config.txt`
Human-readable summary of all choices for documentation and debugging.

### 3. Copy-paste command
A single line that users can run locally without the GUI.

---

## Terminology Mapping (Scientist → ML)

| Scientist term | ML/GUI term |
|----------------|-------------|
| Trial condition | Phase / Genotype |
| Stimulus type | Stim values |
| Lick / response | Response values |
| Brain regions | Brain areas (tokens) |
| Time bins | Timepoints (compressed per token) |
| Train vs. test | Train vs. validation split |

---

## Validation Rules

- Dataset path must exist and contain `processed_data` or `T`
- Phase 1 and Phase 2 must be different for phase classification
- Batch size must be even (for balanced sampling)
- Output directory must be writable

---

## Dependencies

- MATLAB R2016a+ (for `uifigure` / `ui.*` components)
- Fallback: If `uifigure` unavailable, use `figure` + `uicontrol` (classic GUIDE-style)

---

## File Structure

```
prismt/
├── gui/
│   ├── prismt_training_setup.m    # Main GUI entry point
│   └── generate_training_script.m  # Script generation logic
├── train.py                        # Existing training script
└── docs/
    └── PRISMT_GUI_DESIGN.md        # This document
```

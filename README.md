# PRISMT: Data Standardization Pipeline

PRISMT (Pipeline for Research In Standardized Modeling and Training) provides tools for standardizing widefield calcium imaging data into a unified MATLAB format.

## Overview

PRISMT standardizes raw widefield and CDKL5 data into a consistent format where **trials are represented as rows** and **variables are represented as fields** that can serve as conditions for comparison or target variables for unique value extraction.

## Standardized Dataset Structure

The standardized `.mat` file contains a structure where:

- **Trials**: Each trial is represented as a row in the dataset
- **Variables**: Each variable is a field that can be:
  - **Conditions**: Used for comparison (e.g., phase, stimulus, response)
  - **Targets**: Variables to extract unique values from (e.g., mouse IDs, genotype labels)

### Data Structure

```matlab
standardized_data
├── n_datasets: scalar integer
└── dataset_XXX: struct (one per dataset, XXX = 001, 002, ...)
    ├── dff: double array (trials, timepoints, brain_areas)
    ├── zscore: double array (trials, timepoints, brain_areas)
    ├── stim: double array (trials, 1)           % Condition variable
    ├── response: double array (trials, 1)       % Condition variable
    ├── phase: char array or cell array          % Condition variable
    ├── mouse: char array (trials, 1)           % Target variable (unique IDs)
    ├── label: double scalar (optional)          % Target variable (unique values)
    └── dataset_type: char array ('widefield' or 'cdkl5')
```

### Key Points

- **Trials as rows**: Each row in `dff`, `zscore`, `stim`, `response` represents one trial
- **Condition variables**: `stim`, `response`, `phase` - used for filtering/comparison
- **Target variables**: `mouse`, `label` - used to extract unique values or group trials
- **Neural data**: `dff` and `zscore` contain the actual neural activity (trials × timepoints × brain_areas)

## Quick Start

### 1. Standardize Your Data

```matlab
cd('scripts')
standardize_data('input.mat', 'standardized.mat', 'cdkl5')  % or 'widefield'
```

### 2. Validate Standardized Data

```bash
python scripts/validate_data.py standardized.mat
```

## Input Formats Supported

### Widefield Data
- MATLAB table `T` with columns: `dff`, `zscore`, `stim`, `response`, `phase`, `mouse`
- Each row represents a trial
- Variables (`stim`, `response`, `phase`, `mouse`) are extracted per trial

### CDKL5 Data
- MATLAB structures `cdkl5_m_wt_struct` and/or `cdkl5_m_mut_struct` with `allen_parcels` data
- Continuous time-series data is split into trials (30 timepoints each)
- Each trial becomes a row in the standardized format
- Genotype labels (`label`) are assigned based on source structure

## Standardization Process

The standardization script:

1. **Extracts trials**: Converts input data into trial-based format (rows = trials)
2. **Preserves variables**: Maintains condition variables (`stim`, `response`, `phase`) and target variables (`mouse`, `label`)
3. **Ensures consistency**: All datasets follow the same structure
4. **Handles orientation**: Automatically detects and corrects data orientation

## Usage Examples

### Standardize CDKL5 Data

```matlab
% In MATLAB
cd('scripts')
standardize_data('cdkl5_raw.mat', 'standardized_cdkl5.mat', 'cdkl5')

% Load and inspect
load('standardized_cdkl5.mat')
fprintf('Number of datasets: %d\n', standardized_data.n_datasets)

% Access first dataset
ds1 = standardized_data.dataset_001;
fprintf('Number of trials: %d\n', size(ds1.dff, 1))
fprintf('Unique mice: %s\n', unique(ds1.mouse))
fprintf('Unique labels: %s\n', mat2str(unique(ds1.label)))
```

### Standardize Widefield Data

```matlab
% In MATLAB
cd('scripts')
standardize_data('widefield_raw.mat', 'standardized_widefield.mat', 'widefield')

% Load and inspect
load('standardized_widefield.mat')
ds1 = standardized_data.dataset_001;
fprintf('Number of trials: %d\n', size(ds1.dff, 1))
fprintf('Unique phases: %s\n', unique(ds1.phase))
fprintf('Unique stimuli: %s\n', mat2str(unique(ds1.stim)))
```

## Working with Standardized Data

### Extract Unique Values from Target Variables

```matlab
% Get unique mouse IDs
unique_mice = unique(ds1.mouse);

% Get unique labels (for CDKL5)
unique_labels = unique(ds1.label);

% Get unique phases (for widefield)
unique_phases = unique(ds1.phase);
```

### Filter by Condition Variables

```matlab
% Filter trials by phase
early_trials = strcmp(ds1.phase, 'early');

% Filter trials by stimulus
stim1_trials = ds1.stim == 1;

% Filter trials by response
response_trials = ds1.response == 1;

% Combined filter
selected_trials = early_trials & stim1_trials & response_trials;
```

### Access Neural Data for Specific Trials

```matlab
% Get neural data for selected trials
neural_data = ds1.dff(selected_trials, :, :);  % (n_selected_trials, timepoints, brain_areas)
```

## Project Structure

```
prismt/
├── scripts/
│   ├── standardize_data.m  # MATLAB standardization script
│   └── validate_data.py   # Python validation script
├── data/                   # Data loading modules
├── models/                 # Model architectures
├── training/               # Training utilities
├── utils/                  # Utility functions
└── README.md              # This file
```

## Documentation

**📖 [GitHub Wiki](https://github.com/josueortc/prismt/wiki)** - Comprehensive guide on data standardization:
- Data standardization overview
- Preparing widefield data
- Preparing CDKL5 data
- Standardized data format specification
- Validation and troubleshooting

## Key Features

- **Trial-based structure**: Rows represent trials, making it easy to filter and group
- **Variable organization**: Clear separation between condition and target variables
- **Unified format**: Consistent structure for both widefield and CDKL5 data
- **Validation**: Built-in validation script to verify data structure
- **Error handling**: Robust error checking and NaN handling

## Installation

```bash
git clone https://github.com/josueortc/prismt.git
cd prismt
pip install -r requirements.txt
```

## Citation

```bibtex
@software{prismt,
  title = {PRISMT: Unified Pipeline for Widefield and CDKL5 Data},
  author = {Ortega Caro, Josue},
  year = {2024},
  url = {https://github.com/josueortc/prismt}
}
```

## License

[Add your license information here]

## Contact

For questions or issues, please open an issue on GitHub or visit the [Wiki](https://github.com/josueortc/prismt/wiki).

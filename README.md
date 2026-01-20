# PRISMT: Data Standardization Pipeline

PRISMT (Pipeline for Research In Standardized Modeling and Training) provides tools for standardizing widefield calcium imaging data into a unified MATLAB format.

## Overview

PRISMT standardizes raw widefield and CDKL5 data into a consistent format that can be used for training transformer models. The standardization process ensures data consistency, handles different input formats, and prepares data for downstream analysis.

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

## Documentation

**📖 [GitHub Wiki](https://github.com/josueortc/prismt/wiki)** - Comprehensive guide on data standardization:
- Data standardization overview
- Preparing widefield data
- Preparing CDKL5 data
- Standardized data format specification
- Validation and troubleshooting

## Installation

```bash
git clone https://github.com/josueortc/prismt.git
cd prismt
pip install -r requirements.txt
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

## Data Standardization

The standardization process converts raw data into a unified format:

### Input Formats Supported

- **Widefield Data**: MATLAB table `T` with columns: `dff`, `zscore`, `stim`, `response`, `phase`, `mouse`
- **CDKL5 Data**: MATLAB structures `cdkl5_m_wt_struct` and/or `cdkl5_m_mut_struct` with `allen_parcels` data

### Output Format

All standardized data follows this structure:

```matlab
standardized_data
├── n_datasets: scalar integer
└── dataset_XXX: struct
    ├── dff: (trials, timepoints, brain_areas)
    ├── zscore: (trials, timepoints, brain_areas)
    ├── stim: (trials, 1)
    ├── response: (trials, 1)
    ├── phase: char array
    ├── mouse: char array
    ├── label: scalar (optional, for CDKL5)
    └── dataset_type: 'widefield' or 'cdkl5'
```

## Usage Examples

### Standardize CDKL5 Data

```matlab
% In MATLAB
cd('scripts')
standardize_data('cdkl5_raw.mat', 'standardized_cdkl5.mat', 'cdkl5')
```

### Standardize Widefield Data

```matlab
% In MATLAB
cd('scripts')
standardize_data('widefield_raw.mat', 'standardized_widefield.mat', 'widefield')
```

### Validate Standardized Data

```bash
python scripts/validate_data.py standardized.mat
```

## Key Features

- **Unified Format**: Consistent structure for both widefield and CDKL5 data
- **Automatic Detection**: Handles different data orientations and formats
- **Validation**: Built-in validation script to verify data structure
- **Error Handling**: Robust error checking and NaN handling

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

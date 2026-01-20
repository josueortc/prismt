# PRISMT: Unified Pipeline for Widefield and CDKL5 Data

PRISMT (Pipeline for Research In Standardized Modeling and Training) provides a unified interface for training transformer models on widefield calcium imaging data, supporting both standard widefield experiments and CDKL5 genotype classification tasks.

## Quick Start

1. **Standardize your data** (see [WIKI.md](WIKI.md) for detailed instructions):
   ```matlab
   cd('scripts')
   standardize_data('input.mat', 'standardized.mat', 'cdkl5')  % or 'widefield'
   ```

2. **Validate standardized data**:
   ```bash
   python scripts/validate_data.py standardized.mat
   ```

3. **Train model**:
   ```bash
   python train.py --data_path standardized.mat --task_type genotype --epochs 100
   ```

## Features

- **Unified Data Format**: Standardized MATLAB data structure for both widefield and CDKL5 data
- **Automatic Task Detection**: Automatically detects whether data is for phase classification or genotype classification
- **Flexible Training**: Single training script handles both task types
- **Data Validation**: Built-in validation script to verify data structure
- **Modular Architecture**: Clean separation of data loading, models, and training

## Installation

```bash
git clone https://github.com/josueortc/prismt.git
cd prismt
pip install -r requirements.txt
```

## Documentation

- **[WIKI.md](WIKI.md)**: Comprehensive guide on preparing `.mat` files for PRISMT
  - Data standardization process
  - Widefield data preparation
  - CDKL5 data preparation
  - Validation and troubleshooting
  - Complete workflow examples

## Project Structure

```
prismt/
├── data/                    # Data loading modules
├── models/                 # Model architectures
├── training/               # Training utilities
├── utils/                  # Utility functions
├── scripts/                # Utility scripts
│   ├── standardize_data.m  # MATLAB standardization script
│   └── validate_data.py   # Python validation script
├── train.py               # Unified training script
├── WIKI.md               # Data preparation guide (START HERE!)
└── README.md             # This file
```

## Usage

### CDKL5 Genotype Classification

```bash
# 1. Standardize data (in MATLAB)
matlab -nodisplay -nosplash -r "cd('scripts'); standardize_data('input.mat', 'output.mat', 'cdkl5'); exit"

# 2. Validate
python scripts/validate_data.py output.mat

# 3. Train
python train.py --data_path output.mat --task_type genotype --epochs 100
```

### Widefield Phase Classification

```bash
# 1. Standardize data (in MATLAB)
matlab -nodisplay -nosplash -r "cd('scripts'); standardize_data('input.mat', 'output.mat', 'widefield'); exit"

# 2. Validate
python scripts/validate_data.py output.mat

# 3. Train
python train.py --data_path output.mat --task_type phase --phase1 early --phase2 late --epochs 100
```

## Training Options

See `python train.py --help` for all options. Key parameters:

- `--data_path`: Path to standardized .mat file (required)
- `--task_type`: 'auto', 'genotype', or 'phase'
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 16)
- `--learning_rate`: Learning rate (default: 5e-5)

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

For questions or issues, please open an issue on GitHub.

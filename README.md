# PRISMT: Unified Pipeline for Widefield and CDKL5 Data

PRISMT (Pipeline for Research In Standardized Modeling and Training) provides a unified interface for training transformer models on widefield calcium imaging data, supporting both standard widefield experiments and CDKL5 genotype classification tasks.

## Features

- **Unified Data Format**: Standardized MATLAB data structure for both widefield and CDKL5 data
- **Automatic Task Detection**: Automatically detects whether data is for phase classification or genotype classification
- **Flexible Training**: Single training script handles both task types
- **Data Validation**: Built-in validation script to verify data structure
- **Modular Architecture**: Clean separation of data loading, models, and training

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/josueortc/prismt.git
   cd prismt
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Standardize Your Data

First, standardize your raw data into the unified format:

**For Widefield Data**:
```matlab
standardize_data('raw_widefield_data.mat', 'standardized_data.mat', 'widefield')
```

**For CDKL5 Data**:
```matlab
standardize_data('raw_cdkl5_data.mat', 'standardized_data.mat', 'cdkl5')
```

### 2. Validate Data Structure

Verify that your standardized data is correct:
```bash
python scripts/validate_data.py standardized_data.mat
```

### 3. Train Model

**Automatic Task Detection**:
```bash
python train.py --data_path standardized_data.mat --epochs 100
```

**Explicit Task Type**:
```bash
# For genotype classification (CDKL5)
python train.py --data_path standardized_data.mat --task_type genotype --epochs 100

# For phase classification (widefield)
python train.py --data_path standardized_data.mat --task_type phase --phase1 early --phase2 late --epochs 100
```

## Data Standardization

The standardization script (`scripts/standardize_data.m`) converts raw data into a unified format:

### Standardized Data Structure

```matlab
standardized_data
├── n_datasets: Number of datasets
└── dataset_XXX: Each dataset contains
    ├── dff: Neural data (trials, timepoints, brain_areas)
    ├── zscore: Z-scored data (trials, timepoints, brain_areas)
    ├── stim: Stimulus values (trials, 1)
    ├── response: Response values (trials, 1)
    ├── phase: Phase string or array
    ├── mouse: Mouse ID string
    ├── label: Classification label (for CDKL5: 0=WT, 1=Mutant)
    └── dataset_type: 'widefield' or 'cdkl5'
```

### Widefield Data

The standardization script handles:
- Table structures (`T`) with columns: dff, zscore, stim, response, phase, mouse
- Preprocessed data structures
- Automatic shape detection and transposition

### CDKL5 Data

The standardization script handles:
- `cdkl5_m_wt_struct` and `cdkl5_m_mut_struct` structures
- `allen_parcels` data extraction
- Trial segmentation (30 timepoints per trial)
- Genotype labeling (WT=0, Mutant=1)
- NaN replacement with 0

## Training Options

### Command Line Arguments

**Data Arguments**:
- `--data_path`: Path to standardized .mat file (required)
- `--data_type`: Type of neural data ('dff' or 'zscore', default: 'dff')
- `--task_type`: Task type ('auto', 'genotype', or 'phase', default: 'auto')
- `--phase1`: First phase for phase classification (required if task_type='phase')
- `--phase2`: Second phase for phase classification (required if task_type='phase')

**Training Parameters**:
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 16)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--weight_decay`: Weight decay (default: 1e-3)
- `--val_split`: Validation split fraction (default: 0.2)

**Model Parameters**:
- `--hidden_dim`: Hidden dimension (default: 128)
- `--num_heads`: Number of attention heads (default: 4)
- `--num_layers`: Number of transformer layers (default: 3)
- `--ff_dim`: Feed-forward dimension (default: 256)
- `--dropout`: Dropout rate (default: 0.3)

**Scheduler Parameters**:
- `--scheduler_type`: Scheduler type ('cosine_warmup', 'cosine', 'reduce_on_plateau', 'step', default: 'cosine_warmup')
- `--warmup_epochs`: Number of warmup epochs (default: 5)
- `--cosine_t_0`: Initial restart period for cosine annealing (default: 10)
- `--cosine_t_mult`: Factor to increase restart period (default: 2)
- `--cosine_eta_min`: Minimum learning rate (default: 1e-6)

**Other Parameters**:
- `--device`: Device to use ('auto', 'cpu', 'cuda', default: 'auto')
- `--seed`: Random seed (default: 42)
- `--save_dir`: Directory to save results (default: 'results')
- `--wandb_project`: WandB project name (default: 'prismt')
- `--wandb_entity`: WandB entity name (default: 'josueortc')
- `--no_wandb`: Disable WandB logging

## Project Structure

```
prismt/
├── data/                    # Data loading modules
│   └── data_loader.py      # WidefieldDataset, TaskDefinition, etc.
├── models/                 # Model architectures
│   └── transformer.py     # WidefieldTransformer
├── training/               # Training utilities
│   └── trainer.py         # Trainer class
├── utils/                  # Utility functions
│   └── helpers.py         # Helper functions
├── scripts/                # Utility scripts
│   ├── standardize_data.m  # MATLAB standardization script
│   └── validate_data.py   # Python validation script
├── train.py               # Unified training script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Examples

### Example 1: CDKL5 Genotype Classification

```bash
# 1. Standardize data
matlab -nodisplay -nosplash -r "standardize_data('raw_cdkl5.mat', 'standardized.mat', 'cdkl5'); exit"

# 2. Validate
python scripts/validate_data.py standardized.mat

# 3. Train
python train.py --data_path standardized.mat --task_type genotype --epochs 100 --batch_size 16
```

### Example 2: Widefield Phase Classification

```bash
# 1. Standardize data
matlab -nodisplay -nosplash -r "standardize_data('raw_widefield.mat', 'standardized.mat', 'widefield'); exit"

# 2. Validate
python scripts/validate_data.py standardized.mat

# 3. Train
python train.py --data_path standardized.mat --task_type phase --phase1 early --phase2 late --epochs 100
```

## Data Validation

The validation script (`scripts/validate_data.py`) checks:
- Correct data structure
- Required fields present
- Consistent shapes across datasets
- NaN value detection
- Data type consistency

Run validation:
```bash
python scripts/validate_data.py standardized_data.mat
```

## Model Architecture

The transformer model includes:
1. **Token Embedding**: Projects brain area time series to hidden dimensions
2. **CLS Token**: Classification token (similar to BERT)
3. **Positional Embeddings**: Learnable embeddings for brain area positions
4. **Transformer Layers**: Multi-head self-attention + feed-forward networks
5. **Classification Head**: Linear layer on CLS token output

## Task Types

### Genotype Classification (CDKL5)
- **Task**: Binary classification (Wild Type vs Mutant)
- **Labels**: Determined from mouse ID prefixes (`wt_` = 0, `mut_` = 1)
- **Splitting**: Animal-level splitting to prevent data leakage

### Phase Classification (Widefield)
- **Task**: Binary classification between two training phases
- **Labels**: Determined from phase field
- **Splitting**: Mouse-level splitting to prevent data leakage

## Troubleshooting

### Common Issues

1. **Shape Mismatch Errors**
   - Ensure data is standardized correctly
   - Re-run standardization script
   - Check validation script output

2. **NaN Values**
   - Standardization script replaces NaN with 0
   - Check validation script for NaN warnings

3. **Task Type Detection**
   - Use `--task_type` to explicitly specify task type
   - Check mouse ID prefixes for genotype classification

## Citation

If you use PRISMT in your research, please cite:

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

For questions or issues, please open an issue on GitHub or contact the maintainer.

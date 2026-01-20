# PRISMT Testing Guide

## Quick Test Commands

### 1. Test MATLAB Standardization (CDKL5)

```matlab
% In MATLAB
cd('/Users/josueortegacaro/repos/prismt/scripts')
standardize_data('/Users/josueortegacaro/Documents/rachel_cdkl5/cdkl5_data_for_josue_w_states.mat', ...
                 '/tmp/test_cdkl5_standardized.mat', 'cdkl5')
```

Or use the test script:
```matlab
cd('/Users/josueortegacaro/repos/prismt/scripts')
test_standardize
```

### 2. Validate Standardized Data

```bash
cd /Users/josueortegacaro/repos/prismt
python scripts/validate_data.py /tmp/test_cdkl5_standardized.mat
```

### 3. Test Training (Quick Test)

```bash
cd /Users/josueortegacaro/repos/prismt
python train.py \
    --data_path /tmp/test_cdkl5_standardized.mat \
    --task_type genotype \
    --epochs 1 \
    --batch_size 4 \
    --no_wandb
```

## Expected Outputs

### MATLAB Standardization

Expected output:
```
=== PRISMT Data Standardization ===
Input file: /path/to/input.mat
Output file: /path/to/output.mat
Data type: cdkl5

Loading input data...
Processing CDKL5 data...
Processing X wild type animals...
Processing Y mutant animals...
Standardized Z datasets from CDKL5 data
Saving standardized data...
Standardization complete!
Total datasets: Z
```

### Validation Script

Expected output:
```
INFO - Validating standardized data: /path/to/standardized.mat
INFO - Found N datasets
INFO - Dataset 1: type=cdkl5, shape=(trials, 30, 56), mouse=wt_001, ...
...
INFO - === Validation Summary ===
INFO - Total datasets: N
INFO - Dataset types: ['cdkl5']
INFO - Brain area counts: [56]
INFO - Timepoint counts: [30]
INFO - ✓ Validation passed!
```

### Training Script

Expected output:
```
INFO - PRISMT: Unified Training Pipeline
INFO - Loading dataset from: /path/to/standardized.mat
INFO - Auto-detected task type: genotype
INFO - Creating train/validation split...
INFO - Created split: X train, Y val
INFO - Creating data loaders...
INFO - Data dimensions: 28 brain areas, 30 time points
INFO - Creating model...
INFO - Starting training...
...
```

## Troubleshooting

### MATLAB Script Errors

**Error: "Table cell access failed"**
- **Fix**: Already fixed in latest version - handles both cell and matrix storage

**Error: "allen_parcels not found"**
- **Check**: Verify data structure contains `allen_parcels` field
- **Alternative**: Script will skip animals without data

### Validation Script Errors

**Error: "Failed to load v7.3 file"**
- **Fix**: Improved h5py handling in latest version
- **Alternative**: Convert file to v7.0 format in MATLAB: `save('file_v7.mat', 'data', '-v7')`

**Error: "Missing required fields"**
- **Check**: Run MATLAB standardization script first
- **Verify**: Check that standardization completed successfully

### Training Script Errors

**Error: "No valid trials found"**
- **Check**: Verify data was standardized correctly
- **Check**: Verify task_type matches data (genotype vs phase)
- **Fix**: Use explicit `--task_type` argument

**Error: "Shape mismatch"**
- **Check**: Run validation script first
- **Verify**: All datasets have consistent shapes

## Full Pipeline Test

```bash
# 1. Standardize data (in MATLAB)
matlab -nodisplay -nosplash -r "cd('/Users/josueortegacaro/repos/prismt/scripts'); standardize_data('/path/to/input.mat', '/tmp/standardized.mat', 'cdkl5'); exit"

# 2. Validate
python scripts/validate_data.py /tmp/standardized.mat

# 3. Train (quick test)
python train.py --data_path /tmp/standardized.mat --task_type genotype --epochs 1 --batch_size 4 --no_wandb

# 4. Train (full)
python train.py --data_path /tmp/standardized.mat --task_type genotype --epochs 100 --batch_size 16
```

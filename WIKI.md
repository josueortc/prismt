# PRISMT Wiki: Data Preparation Guide

## Table of Contents
1. [Overview](#overview)
2. [Data Standardization Process](#data-standardization-process)
3. [Preparing Widefield Data](#preparing-widefield-data)
4. [Preparing CDKL5 Data](#preparing-cdkl5-data)
5. [Standardized Data Format](#standardized-data-format)
6. [Validation and Troubleshooting](#validation-and-troubleshooting)
7. [Complete Workflow Examples](#complete-workflow-examples)

---

## Overview

PRISMT requires all data to be standardized into a unified MATLAB format before training. This wiki provides comprehensive guidance on preparing your `.mat` files for use with the PRISMT training pipeline.

**Key Points:**
- All data must be standardized using the `standardize_data.m` MATLAB script
- The standardized format is consistent for both widefield and CDKL5 data
- Validation is recommended after standardization
- The standardized file is the input to the training script

---

## Data Standardization Process

### Step 1: Identify Your Data Type

Before standardization, identify which type of data you have:

- **Widefield Data**: Contains a MATLAB table `T` with columns: `dff`, `zscore`, `stim`, `response`, `phase`, `mouse`
- **CDKL5 Data**: Contains MATLAB structures `cdkl5_m_wt_struct` and/or `cdkl5_m_mut_struct` with `allen_parcels` data

### Step 2: Run Standardization Script

The standardization script (`scripts/standardize_data.m`) converts your raw data into the unified format:

```matlab
% Basic usage
standardize_data(input_file, output_file, data_type)

% Examples
standardize_data('raw_widefield.mat', 'standardized.mat', 'widefield')
standardize_data('raw_cdkl5.mat', 'standardized.mat', 'cdkl5')
```

### Step 3: Validate Standardized Data

After standardization, always validate your data:

```bash
python scripts/validate_data.py standardized.mat
```

---

## Preparing Widefield Data

### Input Format

Widefield data should be in one of these formats:

#### Format 1: MATLAB Table `T`
```matlab
% Table T with 6 columns:
T.dff       % Neural data: cell array of arrays (trials × brain_areas × timepoints) or (trials × timepoints × brain_areas)
T.zscore    % Z-scored data: same format as dff
T.stim      % Stimulus values: cell array of vectors (trials × 1)
T.response  % Response values: cell array of vectors (trials × 1)
T.phase     % Phase strings: cell array of strings or single string per row
T.mouse     % Mouse IDs: cell array of strings or single string per row
```

#### Format 2: Preprocessed Data Structure
```matlab
% Already preprocessed structure with 'processed_data' field
processed_data
├── n_datasets: number
└── dataset_XXX: struct arrays with dff, zscore, stim, response, phase, mouse
```

### Standardization Process

The script automatically:
1. **Detects data orientation**: Automatically transposes if needed (brain_areas × timepoints → timepoints × brain_areas)
2. **Handles shape variations**: Supports both `(trials, brain_areas, timepoints)` and `(trials, timepoints, brain_areas)` formats
3. **Preserves metadata**: Keeps all stimulus, response, phase, and mouse information
4. **Adds dataset type**: Marks each dataset as `'widefield'`

### Example: Standardizing Widefield Data

```matlab
% In MATLAB
cd('/path/to/prismt/scripts')

% Standardize widefield table data
standardize_data('/path/to/tableForModeling.mat', ...
                 '/path/to/standardized_widefield.mat', ...
                 'widefield')

% Check output
load('/path/to/standardized_widefield.mat')
fprintf('Standardized %d datasets\n', standardized_data.n_datasets)
```

### Expected Output

After standardization, you should see:
```
=== PRISMT Data Standardization ===
Input file: /path/to/tableForModeling.mat
Output file: /path/to/standardized_widefield.mat
Data type: widefield

Loading input data...
Processing widefield data...
Found table T with N rows
  Dataset 1: dff shape = [trials, timepoints, brain_areas], X trials
  Dataset 2: dff shape = [trials, timepoints, brain_areas], Y trials
  ...
Saving standardized data...
Standardization complete!
Total datasets: N
```

---

## Preparing CDKL5 Data

### Input Format

CDKL5 data must contain one or both of these MATLAB structures:

```matlab
% Wild type animals structure
cdkl5_m_wt_struct  % Struct array or table with fields:
                   % - allen_parcels: (brain_areas × timepoints) or (timepoints × brain_areas)
                   % - mouse: (optional) mouse ID
                   % - Other fields are optional

% Mutant animals structure  
cdkl5_m_mut_struct % Same format as wild type
```

**Key Requirements:**
- `allen_parcels` field is required (contains neural data)
- Data can be stored as struct array or table
- `allen_parcels` can be in either orientation (script auto-detects)
- Mouse IDs are optional (will be auto-generated if missing)

### Standardization Process

The script performs these operations:

1. **Loads CDKL5 structures**: Reads `cdkl5_m_wt_struct` and/or `cdkl5_m_mut_struct`

2. **Extracts `allen_parcels` data**: 
   - Handles both struct arrays and tables
   - Handles both cell arrays and direct matrices
   - Replaces NaN values with 0

3. **Detects and corrects orientation**:
   - Checks if data is `(brain_areas, timepoints)` or `(timepoints, brain_areas)`
   - Uses heuristics: if smaller dimension is 50-100 and larger is 5x+ bigger, smaller = brain_areas
   - Transposes to `(timepoints, brain_areas)` format

4. **Verifies brain area count**:
   - Expects 56 brain areas for CDKL5 data
   - Warns if different count is detected

5. **Splits into trials**:
   - Divides continuous time-series into non-overlapping trials
   - Each trial = 30 timepoints
   - Reshapes to `(n_trials, 30, 56)` format

6. **Creates metadata**:
   - Generates mouse IDs: `wt_001`, `wt_002`, ... for wild type; `mut_001`, `mut_002`, ... for mutants
   - Sets `stim` and `response` to all ones (dummy values)
   - Sets `phase` to `'all'`
   - Sets `label`: 0 for wild type, 1 for mutants
   - Sets `dataset_type` to `'cdkl5'`

### Example: Standardizing CDKL5 Data

```matlab
% In MATLAB
cd('/path/to/prismt/scripts')

% Standardize CDKL5 data
standardize_data('/path/to/cdkl5_data_for_josue_w_states.mat', ...
                 '/path/to/standardized_cdkl5.mat', ...
                 'cdkl5')

% Check output
load('/path/to/standardized_cdkl5.mat')
fprintf('Standardized %d datasets\n', standardized_data.n_datasets)

% Inspect first dataset
fprintf('First dataset shape: %s\n', mat2str(size(standardized_data.dataset_001.dff)))
fprintf('First dataset mouse: %s\n', standardized_data.dataset_001.mouse)
fprintf('First dataset label: %d\n', standardized_data.dataset_001.label)
```

### Expected Output

After standardization, you should see:
```
=== PRISMT Data Standardization ===
Input file: /path/to/cdkl5_data_for_josue_w_states.mat
Output file: /path/to/standardized_cdkl5.mat
Data type: cdkl5

Loading input data...
Processing CDKL5 data...
Processing X wild type animals...
  Animal 1: Found allen_parcels, shape: [56, 27450]
  Animal 2: Found allen_parcels, shape: [56, 28320]
  ...
Processing Y mutant animals...
  Animal 1: Found allen_parcels, shape: [56, 26580]
  Animal 2: Found allen_parcels, shape: [56, 27120]
  ...
Standardized Z datasets from CDKL5 data
Saving standardized data...
Standardization complete!
Total datasets: Z
```

### Important Notes for CDKL5 Data

1. **Brain Areas**: The script expects 56 brain areas. If your data has a different count, you'll see warnings but processing will continue.

2. **Trial Length**: Fixed at 30 timepoints per trial. Any remaining timepoints after division are discarded.

3. **NaN Handling**: All NaN values are automatically replaced with 0 during standardization.

4. **Genotype Labels**: 
   - Wild type animals get `label = 0`
   - Mutant animals get `label = 1`
   - Labels are determined by which structure the animal came from

5. **Mouse ID Format**: 
   - Wild type: `wt_001`, `wt_002`, etc.
   - Mutant: `mut_001`, `mut_002`, etc.
   - If original data has mouse IDs, they're preserved with prefix added

---

## Standardized Data Format

### Structure Overview

All standardized data follows this structure:

```matlab
standardized_data
├── n_datasets: scalar integer
└── dataset_XXX: struct (one per dataset, XXX = 001, 002, ...)
    ├── dff: double array (trials, timepoints, brain_areas)
    ├── zscore: double array (trials, timepoints, brain_areas)
    ├── stim: double array (trials, 1)
    ├── response: double array (trials, 1)
    ├── phase: char array or cell array of strings
    ├── mouse: char array (mouse ID string)
    ├── label: double scalar (optional, for CDKL5: 0=WT, 1=Mutant)
    └── dataset_type: char array ('widefield' or 'cdkl5')
```

### Field Descriptions

#### `dff` and `zscore`
- **Shape**: `(trials, timepoints, brain_areas)`
- **Type**: `double` array
- **Content**: Neural activity data
- **Note**: For CDKL5, `zscore` is identical to `dff` (no separate z-scoring performed)

#### `stim` and `response`
- **Shape**: `(trials, 1)`
- **Type**: `double` array
- **Content**: 
  - Widefield: Actual stimulus/response values from experiments
  - CDKL5: All ones (dummy values, not used for genotype classification)

#### `phase`
- **Type**: `char` array or `cell` array of strings
- **Content**:
  - Widefield: Training phase ('early', 'mid', 'late', etc.)
  - CDKL5: Always `'all'`

#### `mouse`
- **Type**: `char` array
- **Content**: Unique mouse identifier
- **Format**:
  - Widefield: Original mouse IDs from data
  - CDKL5: `wt_XXX` or `mut_XXX` format

#### `label` (CDKL5 only)
- **Type**: `double` scalar
- **Content**: 
  - `0` = Wild type
  - `1` = Mutant
- **Note**: Only present for CDKL5 data

#### `dataset_type`
- **Type**: `char` array
- **Content**: `'widefield'` or `'cdkl5'`
- **Purpose**: Identifies the source data type

### Shape Requirements

**Widefield Data:**
- `dff`/`zscore`: `(trials, timepoints, brain_areas)`
  - Typical: `(N, 41, 82)` before averaging, `(N, 41, 41)` after averaging
- `stim`/`response`: `(trials, 1)`

**CDKL5 Data:**
- `dff`/`zscore`: `(trials, 30, 56)`
  - After averaging in Python: `(trials, 30, 28)`
- `stim`/`response`: `(trials, 1)` (all ones)

### MATLAB File Format

The standardized data is saved as MATLAB v7.3 format (HDF5-based) for compatibility:

```matlab
save('standardized.mat', 'standardized_data', '-v7.3')
```

This format:
- Supports large files (>2GB)
- Is readable by both MATLAB and Python (via `h5py`)
- Preserves all data types and structures

---

## Validation and Troubleshooting

### Running Validation

Always validate your standardized data before training:

```bash
cd /path/to/prismt
python scripts/validate_data.py standardized.mat
```

### What Validation Checks

The validation script verifies:

1. **File Structure**:
   - File exists and is readable
   - Contains `standardized_data` structure
   - Has `n_datasets` field

2. **Required Fields**:
   - Each dataset has: `dff`, `zscore`, `stim`, `response`, `phase`, `mouse`, `dataset_type`
   - Optional: `label` (for CDKL5)

3. **Data Shapes**:
   - `dff` and `zscore` are 3D arrays
   - `stim` and `response` match number of trials
   - Consistent shapes across datasets

4. **Data Quality**:
   - NaN value detection
   - Shape consistency warnings
   - Type consistency

### Common Validation Errors

#### Error: "File not found"
**Solution**: Check the file path is correct

#### Error: "No datasets found"
**Solution**: Re-run standardization script, check input data format

#### Error: "Missing required fields"
**Solution**: Re-run standardization script, ensure input data has all required fields

#### Warning: "Inconsistent brain area counts"
**Solution**: 
- For CDKL5: Check that all animals have 56 brain areas
- For widefield: Check data preprocessing
- Re-run standardization if needed

#### Warning: "Found NaN values"
**Solution**: 
- For CDKL5: NaN replacement should happen automatically
- Check standardization script logs
- Re-run standardization if NaN count is high

### Troubleshooting Standardization

#### Issue: "Table cell access failed"
**Solution**: Already fixed in latest version - handles both cell arrays and matrices

#### Issue: "allen_parcels not found"
**Solution**: 
- Verify your CDKL5 data contains `allen_parcels` field
- Check field name spelling (case-sensitive)
- Check if data is in struct array vs table format

#### Issue: "Wrong number of brain areas"
**Solution**:
- For CDKL5: Verify data has 56 brain areas
- Check data orientation (may need manual transposition)
- Review standardization script logs for warnings

#### Issue: "No trials created"
**Solution**:
- Check that timepoints > 30
- Verify data orientation is correct
- Check for empty or corrupted data

---

## Complete Workflow Examples

### Example 1: CDKL5 Genotype Classification

```matlab
%% Step 1: Standardize CDKL5 Data (MATLAB)
cd('/path/to/prismt/scripts')

input_file = '/path/to/cdkl5_data_for_josue_w_states.mat';
output_file = '/path/to/standardized_cdkl5.mat';

standardize_data(input_file, output_file, 'cdkl5');

fprintf('Standardization complete!\n');
```

```bash
# Step 2: Validate Data (Python)
cd /path/to/prismt
python scripts/validate_data.py /path/to/standardized_cdkl5.mat
```

```bash
# Step 3: Train Model (Python)
python train.py \
    --data_path /path/to/standardized_cdkl5.mat \
    --task_type genotype \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 5e-5
```

### Example 2: Widefield Phase Classification

```matlab
%% Step 1: Standardize Widefield Data (MATLAB)
cd('/path/to/prismt/scripts')

input_file = '/path/to/tableForModeling.mat';
output_file = '/path/to/standardized_widefield.mat';

standardize_data(input_file, output_file, 'widefield');

fprintf('Standardization complete!\n');
```

```bash
# Step 2: Validate Data (Python)
cd /path/to/prismt
python scripts/validate_data.py /path/to/standardized_widefield.mat
```

```bash
# Step 3: Train Model (Python)
python train.py \
    --data_path /path/to/standardized_widefield.mat \
    --task_type phase \
    --phase1 early \
    --phase2 late \
    --epochs 100 \
    --batch_size 16
```

### Example 3: Batch Processing Multiple Files

```matlab
%% Batch standardize multiple CDKL5 files
cd('/path/to/prismt/scripts')

input_files = {
    '/data/cdkl5_batch1.mat'
    '/data/cdkl5_batch2.mat'
    '/data/cdkl5_batch3.mat'
};

for i = 1:length(input_files)
    input_file = input_files{i};
    [~, name, ~] = fileparts(input_file);
    output_file = sprintf('/data/standardized_%s.mat', name);
    
    fprintf('Processing %s...\n', input_file);
    standardize_data(input_file, output_file, 'cdkl5');
    fprintf('Saved to %s\n\n', output_file);
end
```

---

## Best Practices

### Before Standardization

1. **Backup your data**: Always keep original files
2. **Check data format**: Verify your data matches expected input format
3. **Check file size**: Ensure files are not corrupted
4. **Review data structure**: Use MATLAB's `whos` command to inspect structure

### During Standardization

1. **Monitor output**: Watch for warnings about brain area counts or data orientation
2. **Check logs**: Review standardization script output for errors
3. **Verify output**: Load and inspect standardized data in MATLAB

### After Standardization

1. **Always validate**: Run validation script before training
2. **Check shapes**: Verify data shapes match expectations
3. **Inspect samples**: Look at a few datasets to ensure data looks correct
4. **Document**: Note any special handling or data transformations

### Data Quality Checks

- **Consistency**: All datasets should have same brain area and timepoint counts
- **Completeness**: No missing fields or empty datasets
- **Cleanliness**: No NaN values (should be replaced with 0)
- **Correctness**: Mouse IDs, labels, and metadata are accurate

---

## Advanced Topics

### Custom Data Formats

If your data doesn't match the expected formats, you may need to:

1. **Pre-process in MATLAB**: Convert your format to one of the supported formats
2. **Modify standardization script**: Add support for your specific format
3. **Contact maintainers**: Request support for new data formats

### Large Files

For very large files (>10GB):

1. **Use v7.3 format**: Already default, supports large files
2. **Process in batches**: Standardize subsets separately
3. **Check disk space**: Ensure enough space for output file
4. **Monitor memory**: Large files may require significant RAM

### Performance Optimization

1. **Pre-allocate arrays**: If modifying script, pre-allocate output arrays
2. **Vectorize operations**: Use MATLAB vectorization for speed
3. **Parallel processing**: Consider parallelizing across animals/datasets
4. **Memory management**: Clear large variables when done

---

## Summary

**Key Takeaways:**

1. ✅ Always standardize data before training
2. ✅ Use `standardize_data.m` for both widefield and CDKL5 data
3. ✅ Validate standardized data with `validate_data.py`
4. ✅ Check standardization logs for warnings
5. ✅ Verify data shapes and consistency
6. ✅ Keep original data files as backup

**Quick Reference:**

```matlab
% Standardize
standardize_data('input.mat', 'output.mat', 'cdkl5')  % or 'widefield'
```

```bash
# Validate
python scripts/validate_data.py output.mat

# Train
python train.py --data_path output.mat --task_type genotype
```

For more information, see the main [README.md](README.md) file.

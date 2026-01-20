#!/usr/bin/env python3
"""
Validate standardized data structure for PRISMT training pipeline.

This script checks that the standardized .mat file has the correct structure
and data formats for training.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import scipy.io as sio
import h5py
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_standardized_data(mat_file_path: str) -> bool:
    """
    Validate standardized data structure.
    
    Args:
        mat_file_path: Path to standardized .mat file
        
    Returns:
        True if validation passes, False otherwise
    """
    logger.info(f"Validating standardized data: {mat_file_path}")
    
    mat_path = Path(mat_file_path)
    if not mat_path.exists():
        logger.error(f"File not found: {mat_file_path}")
        return False
    
    # Load data
    try:
        try:
            mat_data = sio.loadmat(str(mat_path), struct_as_record=False)
            standardized_data = mat_data['standardized_data']
        except NotImplementedError:
            # Use h5py for v7.3 files
            with h5py.File(str(mat_path), 'r') as f:
                standardized_data_ref = f['standardized_data']
                n_datasets = int(standardized_data_ref['n_datasets'][0, 0])
                standardized_data = {'n_datasets': n_datasets}
                
                for i in range(1, n_datasets + 1):
                    dataset_name = f'dataset_{i:03d}'
                    if dataset_name in standardized_data_ref:
                        dataset_ref = standardized_data_ref[dataset_name]
                        standardized_data[dataset_name] = {}
                        for field in ['dff', 'zscore', 'stim', 'response', 'phase', 'mouse', 'label', 'dataset_type']:
                            if field in dataset_ref:
                                field_ref = dataset_ref[field]
                                if hasattr(field_ref, '__getitem__'):
                                    standardized_data[dataset_name][field] = field_ref[()]
                                else:
                                    standardized_data[dataset_name][field] = field_ref
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return False
    
    # Check structure
    if isinstance(standardized_data, dict):
        n_datasets = standardized_data.get('n_datasets', 0)
    else:
        n_datasets = standardized_data.n_datasets[0, 0] if hasattr(standardized_data, 'n_datasets') else 0
    
    if n_datasets == 0:
        logger.error("No datasets found in standardized_data")
        return False
    
    logger.info(f"Found {n_datasets} datasets")
    
    # Validate each dataset
    all_valid = True
    dataset_types = set()
    brain_area_counts = set()
    timepoint_counts = set()
    
    for i in range(1, n_datasets + 1):
        dataset_name = f'dataset_{i:03d}'
        
        if isinstance(standardized_data, dict):
            dataset = standardized_data.get(dataset_name, {})
        else:
            dataset = getattr(standardized_data, dataset_name, None)
        
        if not dataset:
            logger.warning(f"Dataset {i} ({dataset_name}) not found")
            all_valid = False
            continue
        
        # Check required fields
        required_fields = ['dff', 'zscore', 'stim', 'response', 'phase', 'mouse', 'dataset_type']
        missing_fields = []
        for field in required_fields:
            if isinstance(dataset, dict):
                if field not in dataset:
                    missing_fields.append(field)
            else:
                if not hasattr(dataset, field):
                    missing_fields.append(field)
        
        if missing_fields:
            logger.error(f"Dataset {i}: Missing required fields: {missing_fields}")
            all_valid = False
            continue
        
        # Get field values
        if isinstance(dataset, dict):
            dff = dataset['dff']
            zscore = dataset['zscore']
            stim = dataset['stim']
            response = dataset['response']
            phase = dataset['phase']
            mouse = dataset['mouse']
            dataset_type = dataset.get('dataset_type', 'unknown')
            label = dataset.get('label', None)
        else:
            dff = dataset.dff
            zscore = dataset.zscore
            stim = dataset.stim
            response = dataset.response
            phase = dataset.phase
            mouse = dataset.mouse
            dataset_type = getattr(dataset, 'dataset_type', 'unknown')
            label = getattr(dataset, 'label', None)
        
        # Validate dff shape
        if not isinstance(dff, np.ndarray):
            logger.error(f"Dataset {i}: dff is not a numpy array")
            all_valid = False
            continue
        
        if dff.ndim != 3:
            logger.error(f"Dataset {i}: dff must be 3D array, got {dff.ndim}D")
            all_valid = False
            continue
        
        n_trials, n_timepoints, n_brain_areas = dff.shape
        dataset_types.add(dataset_type)
        brain_area_counts.add(n_brain_areas)
        timepoint_counts.add(n_timepoints)
        
        # Check zscore matches dff
        if zscore.shape != dff.shape:
            logger.error(f"Dataset {i}: zscore shape {zscore.shape} doesn't match dff shape {dff.shape}")
            all_valid = False
        
        # Check stim and response
        if stim.shape[0] != n_trials:
            logger.error(f"Dataset {i}: stim length {stim.shape[0]} doesn't match n_trials {n_trials}")
            all_valid = False
        
        if response.shape[0] != n_trials:
            logger.error(f"Dataset {i}: response length {response.shape[0]} doesn't match n_trials {n_trials}")
            all_valid = False
        
        # Check for NaN values
        nan_count_dff = np.sum(np.isnan(dff))
        nan_count_zscore = np.sum(np.isnan(zscore))
        
        if nan_count_dff > 0:
            logger.warning(f"Dataset {i}: Found {nan_count_dff} NaN values in dff")
        
        if nan_count_zscore > 0:
            logger.warning(f"Dataset {i}: Found {nan_count_zscore} NaN values in zscore")
        
        # Log dataset info
        logger.info(f"Dataset {i}: type={dataset_type}, shape={dff.shape}, mouse={mouse}, "
                   f"phase={phase}, label={label}, NaN_dff={nan_count_dff}, NaN_zscore={nan_count_zscore}")
    
    # Summary
    logger.info("\n=== Validation Summary ===")
    logger.info(f"Total datasets: {n_datasets}")
    logger.info(f"Dataset types: {sorted(dataset_types)}")
    logger.info(f"Brain area counts: {sorted(brain_area_counts)}")
    logger.info(f"Timepoint counts: {sorted(timepoint_counts)}")
    
    # Check consistency
    if len(brain_area_counts) > 1:
        logger.warning(f"⚠️  Inconsistent brain area counts: {sorted(brain_area_counts)}")
        logger.warning("   All datasets should have the same number of brain areas")
    
    if len(timepoint_counts) > 1:
        logger.warning(f"⚠️  Inconsistent timepoint counts: {sorted(timepoint_counts)}")
        logger.warning("   All datasets should have the same number of timepoints")
    
    if all_valid:
        logger.info("✓ Validation passed!")
    else:
        logger.error("✗ Validation failed! Check errors above.")
    
    return all_valid


def main():
    parser = argparse.ArgumentParser(description='Validate standardized PRISMT data')
    parser.add_argument('data_path', type=str, help='Path to standardized .mat file')
    
    args = parser.parse_args()
    
    is_valid = validate_standardized_data(args.data_path)
    
    sys.exit(0 if is_valid else 1)


if __name__ == '__main__':
    main()

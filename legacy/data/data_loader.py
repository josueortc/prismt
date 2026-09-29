"""
Data loading utilities for widefield calcium imaging data.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import scipy.io as sio
import h5py
from sklearn.model_selection import GroupKFold
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from collections import Counter
import random

logger = logging.getLogger(__name__)


class BalancedBatchSampler(Sampler):
    """
    Sampler that ensures each batch has equal samples from both classes.
    """
    
    def __init__(self, dataset, batch_size: int, drop_last: bool = True):
        """
        Initialize balanced batch sampler.
        
        Args:
            dataset: WidefieldTrialDataset instance
            batch_size: Total batch size (must be even for balanced sampling)
            drop_last: Whether to drop the last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        if batch_size % 2 != 0:
            raise ValueError("Batch size must be even for balanced sampling")
        
        self.samples_per_class = batch_size // 2
        
        # Group indices by class
        self.class_indices = {0: [], 1: []}
        for idx, label in enumerate(dataset.labels):
            self.class_indices[label].append(idx)
        
        # Log class distribution
        n_class_0 = len(self.class_indices[0])
        n_class_1 = len(self.class_indices[1])
        logger.info(f"Balanced sampler - Class 0: {n_class_0}, Class 1: {n_class_1}")
        
        # Calculate number of batches
        min_samples = min(n_class_0, n_class_1)
        self.num_batches = min_samples // self.samples_per_class
        
        if self.num_batches == 0:
            raise ValueError(f"Not enough samples for balanced batching. "
                           f"Need at least {self.samples_per_class} samples per class, "
                           f"but got {min_samples} for minority class")
        
        logger.info(f"Balanced sampler will create {self.num_batches} balanced batches")
    
    def __iter__(self):
        """Generate balanced batches."""
        # Shuffle indices for each class
        class_0_shuffled = random.sample(self.class_indices[0], len(self.class_indices[0]))
        class_1_shuffled = random.sample(self.class_indices[1], len(self.class_indices[1]))
        
        # Create balanced batches
        for batch_idx in range(self.num_batches):
            batch_indices = []
            
            # Add samples from class 0
            start_idx = batch_idx * self.samples_per_class
            end_idx = start_idx + self.samples_per_class
            batch_indices.extend(class_0_shuffled[start_idx:end_idx])
            
            # Add samples from class 1
            batch_indices.extend(class_1_shuffled[start_idx:end_idx])
            
            # Shuffle the batch to mix classes
            random.shuffle(batch_indices)
            yield batch_indices
    
    def __len__(self):
        """Return number of batches."""
        return self.num_batches


class WidefieldDataset:
    """
    Main dataset class for loading and managing widefield calcium imaging data.
    """
    
    def __init__(self, mat_file_path: str):
        """
        Initialize the dataset by loading the .mat file.
        
        Args:
            mat_file_path: Path to the .mat file containing the data table
        """
        self.mat_file_path = Path(mat_file_path)
        self.data_table = None
        self.load_data()
        
    def load_data(self) -> None:
        """Load data from the .mat file.
        
        Expected MATLAB table structure with columns in order:
        dff, zscore, stim, response, phase, mouse
        
        If a preprocessed file exists (*_processed.mat), it will be used.
        Otherwise, the original table T will be loaded.
        """
        if not self.mat_file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.mat_file_path}")
        
        # Check if this file is already a preprocessed file (contains processed_data)
        # First check if file itself contains processed_data structure
        is_preprocessed = False
        try:
            # Try scipy first
            try:
                mat_data = sio.loadmat(str(self.mat_file_path), struct_as_record=False)
                if 'processed_data' in mat_data:
                    is_preprocessed = True
            except NotImplementedError:
                # MATLAB v7.3 file, use h5py
                with h5py.File(str(self.mat_file_path), 'r') as f:
                    if 'processed_data' in f:
                        is_preprocessed = True
        except Exception:
            pass
        
        if is_preprocessed:
            logger.info(f"File contains processed_data structure, loading as preprocessed file")
            logger.info("Loading preprocessed data...")
            self.data_table = self._load_preprocessed_data(self.mat_file_path)
        else:
            # Check for separate preprocessed file (*_processed.mat)
            preprocessed_path = self.mat_file_path.parent / (self.mat_file_path.stem + '_processed.mat')
            if preprocessed_path.exists():
                logger.info(f"Found preprocessed file: {preprocessed_path}")
                logger.info("Loading preprocessed data...")
                self.data_table = self._load_preprocessed_data(preprocessed_path)
            else:
                logger.warning(f"Preprocessed file not found: {preprocessed_path}")
                logger.info(f"Loading original MATLAB table from {self.mat_file_path}")
                logger.warning("Note: Using fallback loader - metadata identification may fail. Consider preprocessing with utils/preprocess_matlab_table.m for reliable data loading.")
                
                # Try loading with scipy first, then h5py for v7.3 files
                try:
                    mat_data = sio.loadmat(str(self.mat_file_path), struct_as_record=False)
                    if 'T' not in mat_data:
                        raise ValueError("Expected variable 'T' not found in .mat file")
                    
                    # For scipy, T should be a structured array or table
                    T = mat_data['T']
                    
                    # If it's a structured array, convert to our table format
                    if isinstance(T, np.ndarray) and T.dtype.names:
                        # Structured array - columns are named
                        self.data_table = self._load_scipy_table(T)
                    else:
                        # Assume it's already in the right format or needs conversion
                        self.data_table = self._load_scipy_table(T)
                    
                except NotImplementedError:
                    # MATLAB v7.3 file, use h5py
                    logger.info("MATLAB v7.3 file detected, using h5py reader")
                    with h5py.File(str(self.mat_file_path), 'r') as f:
                        if 'T' not in f:
                            raise ValueError("Expected variable 'T' not found in .mat file")
                        
                        # Load the table data - read columns in order: dff, zscore, stim, response, phase, mouse
                        # The simple loader now maps metadata suffixes to columns in expected order
                        self.data_table = self._load_h5py_table_simple(f)
        
        logger.info(f"Loaded data table with shape: {self.data_table.shape}")
    
    def _load_preprocessed_data(self, preprocessed_path: Path):
        """
        Load preprocessed MATLAB data that was created by preprocess_matlab_table.m
        
        Structure:
        - processed_data.n_datasets: number of datasets
        - processed_data.dataset_XXX: each dataset with dff, zscore, stim, response, phase, mouse
        """
        logger.info(f"Loading preprocessed data from {preprocessed_path}")
        
        try:
            # Try scipy first
            mat_data = sio.loadmat(str(preprocessed_path), struct_as_record=False)
            processed_data = mat_data['processed_data']
        except NotImplementedError:
            # Use h5py for v7.3 files
            with h5py.File(str(preprocessed_path), 'r') as f:
                processed_data_ref = f['processed_data']
                # Read the structure
                n_datasets = int(processed_data_ref['n_datasets'][0, 0])
                processed_data = {'n_datasets': n_datasets}
                
                # Read each dataset
                for i in range(1, n_datasets + 1):
                    dataset_name = f'dataset_{i:03d}'
                    if dataset_name in processed_data_ref:
                        dataset_ref = processed_data_ref[dataset_name]
                        # Read fields
                        processed_data[dataset_name] = {}
                        for field in ['dff', 'zscore', 'stim', 'response', 'phase', 'mouse', 'n_trials']:
                            if field in dataset_ref:
                                field_ref = dataset_ref[field]
                                
                                # Read the data - handle references (MATLAB strings/cells are stored as references)
                                # MATLAB strings in v7.3 .mat files are stored as reference arrays
                                try:
                                    # Check if this is a reference dataset (for MATLAB strings)
                                    # Check dtype first before reading
                                    is_reference_dtype = False
                                    if hasattr(field_ref, 'dtype'):
                                        try:
                                            # Check if dtype is a reference type
                                            dtype_str = str(field_ref.dtype)
                                            if 'ref' in dtype_str.lower() or 'reference' in dtype_str.lower():
                                                is_reference_dtype = True
                                            # Also check using h5py special dtype
                                            try:
                                                ref_dtype = h5py.special_dtype(ref=h5py.Reference)
                                                if field_ref.dtype == ref_dtype:
                                                    is_reference_dtype = True
                                            except:
                                                pass
                                        except:
                                            pass
                                    
                                    if is_reference_dtype:
                                        # It's a reference dataset - read the reference and follow it
                                        refs = field_ref[()]
                                        if i <= 3:
                                            logger.debug(f"Dataset {i}: {field} is a reference dataset, refs: {refs}")
                                        
                                        if len(refs) > 0:
                                            # Get the first reference (MATLAB strings are scalar references)
                                            ref_val = refs[0] if isinstance(refs, np.ndarray) and len(refs) > 0 else refs
                                            if ref_val is not None:
                                                try:
                                                    # Follow the reference to get the actual string data
                                                    referenced_dataset = f[ref_val]
                                                    if i <= 3:
                                                        logger.debug(f"Dataset {i}: Following {field} reference {ref_val}, dtype: {referenced_dataset.dtype}")
                                                    
                                                    # Read the referenced string data
                                                    if referenced_dataset.dtype.kind in ['S', 'U']:
                                                        # String dataset - read and decode
                                                        if referenced_dataset.ndim == 2:
                                                            field_data = referenced_dataset[0, :].tobytes().decode('utf-8').strip('\x00') if referenced_dataset.dtype.kind == 'S' else ''.join(referenced_dataset[0, :].flatten())
                                                        else:
                                                            field_data = referenced_dataset.tobytes().decode('utf-8').strip('\x00') if referenced_dataset.dtype.kind == 'S' else str(referenced_dataset.item())
                                                    else:
                                                        # Not a string - try to read as string anyway
                                                        field_data = referenced_dataset[()]
                                                        if isinstance(field_data, np.ndarray):
                                                            field_data = field_data.tobytes().decode('utf-8').strip('\x00') if field_data.dtype.kind == 'S' else str(field_data.item())
                                                        else:
                                                            field_data = str(field_data)
                                                except (KeyError, ValueError, TypeError) as ref_err:
                                                    if i <= 3:
                                                        logger.warning(f"Dataset {i}: Failed to follow {field} reference {ref_val}: {ref_err}")
                                                    field_data = None
                                            else:
                                                field_data = None
                                        else:
                                            field_data = None
                                    elif hasattr(field_ref, '__getitem__'):
                                        # Regular dataset - read the data
                                        field_data = field_ref[()]
                                    else:
                                        field_data = field_ref
                                except (TypeError, ValueError, KeyError, AttributeError) as e:
                                    # If direct read fails, try as reference
                                    try:
                                        if isinstance(field_ref, h5py.Reference):
                                            field_data = f[field_ref][()]
                                        elif hasattr(field_ref, 'ref'):
                                            # Reference attribute
                                            ref_val = field_ref.ref
                                            if ref_val is not None:
                                                field_data = f[ref_val][()]
                                            else:
                                                field_data = None
                                        else:
                                            field_data = field_ref[()]
                                    except Exception as e2:
                                        if i <= 3:
                                            logger.debug(f"Dataset {i}: Failed to read {field} as reference: {e2}")
                                        field_data = field_ref
                                
                                # Special handling for phase and mouse - they should be strings
                                if field in ['phase', 'mouse']:
                                    # MATLAB char arrays are stored as numeric arrays in h5py
                                    # Need to convert them to strings
                                    # Debug: log raw data for first few datasets
                                    if i <= 3:
                                        logger.debug(f"Dataset {i}: Raw {field} data: type={type(field_data)}, shape={getattr(field_data, 'shape', 'N/A')}, dtype={getattr(field_data, 'dtype', 'N/A')}, value={field_data}")
                                    
                                    if isinstance(field_data, np.ndarray):
                                        if field_data.dtype.kind in ['i', 'u', 'f']:
                                            # Numeric array - might be char codes OR h5py reference
                                            # Check if values are very large (likely references) or small (likely char codes)
                                            try:
                                                original_shape = field_data.shape
                                                field_data_flat = field_data.flatten()
                                                max_val = np.max(np.abs(field_data_flat)) if len(field_data_flat) > 0 else 0
                                                
                                                # If max value is very large (> 1000), it's likely a reference, not char codes
                                                if max_val > 1000:
                                                    # This is likely an h5py reference - the first value is the reference ID
                                                    # Try to follow the reference to get the actual string
                                                    if i <= 3:
                                                        logger.debug(f"Dataset {i}: {field} has large numeric values (max={max_val}), treating as reference")
                                                    
                                                    try:
                                                        # The first value in the array is likely the reference ID
                                                        ref_id = int(field_data_flat[0])
                                                        if i <= 3:
                                                            logger.debug(f"Dataset {i}: Trying to follow {field} reference ID: {ref_id}")
                                                        
                                                        # Try to access the referenced dataset
                                                        # References in h5py are typically stored in a special way
                                                        # Try accessing by the reference ID
                                                        try:
                                                            # Try to get the referenced dataset
                                                            # In h5py, references might be stored in #refs# group
                                                            refs_group = f.get('#refs#', None)
                                                            if refs_group is not None:
                                                                # Try to access the reference
                                                                ref_key = str(ref_id)
                                                                if ref_key in refs_group:
                                                                    referenced = refs_group[ref_key]
                                                                    if referenced.dtype.kind in ['S', 'U']:
                                                                        # String dataset
                                                                        if referenced.ndim == 2:
                                                                            field_data = referenced[0, :].tobytes().decode('utf-8').strip('\x00') if referenced.dtype.kind == 'S' else ''.join(referenced[0, :].flatten())
                                                                        else:
                                                                            field_data = referenced.tobytes().decode('utf-8').strip('\x00') if referenced.dtype.kind == 'S' else str(referenced.item())
                                                                    else:
                                                                        field_data = str(referenced.item() if referenced.size == 1 else referenced.flat[0])
                                                                else:
                                                                    raise KeyError(f"Reference {ref_key} not found in #refs#")
                                                            else:
                                                                # No #refs# group, try direct access
                                                                # The reference might be stored differently
                                                                raise KeyError("No #refs# group found")
                                                        except (KeyError, AttributeError, ValueError) as ref_access_err:
                                                            if i <= 3:
                                                                logger.debug(f"Dataset {i}: Could not access reference {ref_id} directly: {ref_access_err}")
                                                            # Try alternative: check if field_ref itself is a reference dataset
                                                            # Re-read field_ref to check its dtype
                                                            if hasattr(field_ref, 'dtype'):
                                                                dtype_str = str(field_ref.dtype)
                                                                if 'ref' in dtype_str.lower() or 'reference' in dtype_str.lower():
                                                                    # It's a reference dataset - read references properly
                                                                    refs = field_ref[()]
                                                                    if len(refs) > 0:
                                                                        ref_val = refs[0] if isinstance(refs, np.ndarray) else refs
                                                                        if ref_val is not None:
                                                                            referenced = f[ref_val]
                                                                            if referenced.dtype.kind in ['S', 'U']:
                                                                                if referenced.ndim == 2:
                                                                                    field_data = referenced[0, :].tobytes().decode('utf-8').strip('\x00') if referenced.dtype.kind == 'S' else ''.join(referenced[0, :].flatten())
                                                                                else:
                                                                                    field_data = referenced.tobytes().decode('utf-8').strip('\x00') if referenced.dtype.kind == 'S' else str(referenced.item())
                                                                            else:
                                                                                field_data = str(referenced.item() if referenced.size == 1 else referenced.flat[0])
                                                                    else:
                                                                        raise ValueError("Empty reference array")
                                                                else:
                                                                    raise ValueError(f"Not a reference dtype: {dtype_str}")
                                                            else:
                                                                raise ValueError("field_ref has no dtype attribute")
                                                    except (ValueError, TypeError, KeyError, AttributeError) as ref_err:
                                                        # Reference reading failed - set to unknown
                                                        if i <= 3:
                                                            logger.warning(f"Dataset {i}: {field} appears to be a reference (max value {max_val}) but couldn't resolve it: {ref_err}. Setting to 'unknown'")
                                                        field_data = 'unknown'
                                                else:
                                                    # Small values - treat as ASCII character codes
                                                    # If it's a 2D array [1, N] or [N, 1], flatten it
                                                    if field_data.ndim == 2:
                                                        if field_data.shape[0] == 1:
                                                            field_data_flat = field_data[0, :].flatten()
                                                        elif field_data.shape[1] == 1:
                                                            field_data_flat = field_data[:, 0].flatten()
                                                        else:
                                                            field_data_flat = field_data.flatten()
                                                    else:
                                                        field_data_flat = field_data.flatten()
                                                    
                                                    # Convert numeric codes to characters
                                                    chars = []
                                                    for c in field_data_flat:
                                                        c_int = int(c)
                                                        if c_int != 0 and (32 <= c_int <= 126 or c_int in [9, 10, 13]):
                                                            try:
                                                                chars.append(chr(c_int))
                                                            except (ValueError, OverflowError):
                                                                pass
                                                    field_data = ''.join(chars).strip()
                                                    
                                                    # Debug: log conversion for first few datasets
                                                    if i <= 3:
                                                        sample_codes = field_data_flat[:min(10, len(field_data_flat))].tolist()
                                                        logger.info(f"Dataset {i}: Converting {field} from shape {original_shape}, sample codes: {sample_codes}, to string: '{field_data}' (length {len(field_data)})")
                                                    # If result is empty, log warning
                                                    if not field_data and i <= 10:
                                                        logger.warning(f"Dataset {i}: {field} conversion resulted in empty string. Raw codes: {field_data_flat.tolist()[:20]}")
                                            except (ValueError, TypeError, OverflowError) as e:
                                                # If conversion fails, try as string
                                                logger.warning(f"Dataset {i}: Failed to convert {field} numeric array: {e}, trying fallback")
                                                field_data = str(field_data.item() if field_data.size == 1 else field_data.flat[0])
                                        elif field_data.dtype.kind == 'S':
                                            # Bytes array - decode it
                                            if field_data.ndim == 2:
                                                # [1, N] or [N, 1] - handle correctly
                                                if field_data.shape[0] == 1:
                                                    field_data = field_data[0, :].tobytes().decode('utf-8').strip('\x00')
                                                elif field_data.shape[1] == 1:
                                                    field_data = field_data[:, 0].tobytes().decode('utf-8').strip('\x00')
                                                else:
                                                    field_data = field_data.flatten().tobytes().decode('utf-8').strip('\x00')
                                            else:
                                                field_data = field_data.tobytes().decode('utf-8').strip('\x00')
                                        elif field_data.dtype.kind == 'U':
                                            # Unicode array
                                            if field_data.ndim == 2:
                                                if field_data.shape[0] == 1:
                                                    field_data = ''.join(field_data[0, :].flatten()).strip()
                                                elif field_data.shape[1] == 1:
                                                    field_data = ''.join(field_data[:, 0].flatten()).strip()
                                                else:
                                                    field_data = ''.join(field_data.flatten()).strip()
                                            else:
                                                field_data = str(field_data.item()).strip()
                                    elif isinstance(field_data, (str, bytes)):
                                        # Already a string or bytes
                                        if isinstance(field_data, bytes):
                                            field_data = field_data.decode('utf-8').strip('\x00').strip()
                                        else:
                                            field_data = str(field_data).strip()
                                    else:
                                        # Other type - convert to string
                                        field_data = str(field_data).strip()
                                    
                                    # Ensure non-empty
                                    if not field_data or field_data == '':
                                        logger.warning(f"Dataset {i}: Empty {field} value, using 'unknown'")
                                        field_data = 'unknown'
                                    
                                    if i <= 3:
                                        logger.info(f"Dataset {i}: Final {field} value: '{field_data}'")
                                
                                processed_data[dataset_name][field] = field_data
        
        # Convert to our table format
        if isinstance(processed_data, dict):
            n_datasets = processed_data.get('n_datasets', len([k for k in processed_data.keys() if k.startswith('dataset_')]))
        else:
            # Structured array
            n_datasets = processed_data['n_datasets'][0, 0] if hasattr(processed_data, 'n_datasets') else 0
        
        data_table = np.empty((n_datasets, 6), dtype=object)
        
        for i in range(n_datasets):
            dataset_name = f'dataset_{i+1:03d}'
            
            if isinstance(processed_data, dict):
                dataset = processed_data.get(dataset_name, {})
            else:
                # Structured array access
                dataset = processed_data[dataset_name]
            
            # Extract data
            dff = dataset['dff'] if isinstance(dataset, dict) else dataset.dff
            zscore = dataset['zscore'] if isinstance(dataset, dict) else dataset.zscore
            stim = dataset['stim'] if isinstance(dataset, dict) else dataset.stim
            response = dataset['response'] if isinstance(dataset, dict) else dataset.response
            phase = dataset['phase'] if isinstance(dataset, dict) else dataset.phase
            mouse = dataset['mouse'] if isinstance(dataset, dict) else dataset.mouse
            
            # Debug: Check for NaN and log shapes for first few datasets
            if i < 3:
                if hasattr(dff, 'shape'):
                    nan_count = np.sum(np.isnan(dff)) if isinstance(dff, np.ndarray) else 0
                    logger.info(f"Dataset {i}: dff RAW shape={dff.shape}, dtype={getattr(dff, 'dtype', 'N/A')}, NaN count={nan_count}/{dff.size if hasattr(dff, 'size') else 'N/A'}")
                    # For 3D arrays, log what we think each dimension represents
                    if isinstance(dff, np.ndarray) and dff.ndim == 3:
                        n1, n2, n3 = dff.shape
                        logger.info(f"Dataset {i}: dff 3D shape analysis: dim0={n1}, dim1={n2}, dim2={n3}")
                        logger.info(f"Dataset {i}: Expected CDKL5 format: (trials, 30, 56)")
                        if n2 == 30 and n3 == 56:
                            logger.info(f"Dataset {i}: ✓ Matches CDKL5 format (trials={n1}, timepoints=30, brain_areas=56)")
                        elif n2 == 56 and n3 == 30:
                            logger.warning(f"Dataset {i}: ⚠️  Wrong orientation: (trials={n1}, brain_areas=56, timepoints=30) - will be transposed")
                        else:
                            logger.warning(f"Dataset {i}: ⚠️  Unexpected shape - may need manual fix")
                if hasattr(zscore, 'shape'):
                    nan_count_z = np.sum(np.isnan(zscore)) if isinstance(zscore, np.ndarray) else 0
                    logger.info(f"Dataset {i}: zscore shape={zscore.shape}, dtype={getattr(zscore, 'dtype', 'N/A')}, NaN count={nan_count_z}/{zscore.size if hasattr(zscore, 'size') else 'N/A'}")
            
            # Debug: Log raw types for first few datasets
            if i < 3:
                logger.debug(f"Dataset {i}: phase type={type(phase)}, shape={getattr(phase, 'shape', 'N/A')}, dtype={getattr(phase, 'dtype', 'N/A')}")
                logger.debug(f"Dataset {i}: mouse type={type(mouse)}, shape={getattr(mouse, 'shape', 'N/A')}, dtype={getattr(mouse, 'dtype', 'N/A')}")
            
            # Ensure correct types and shapes
            if hasattr(stim, 'flatten'):
                stim = stim.flatten().astype(int)
            if hasattr(response, 'flatten'):
                response = response.flatten().astype(int)
            
            # Phase and mouse are single characters/strings per row (not vectors)
            # Convert to string and strip any null/whitespace characters
            # Handle various input types: bytes, numpy arrays, single characters, strings
            # MATLAB char arrays are stored as 2D arrays (e.g., shape [1, N] for strings)
            # Both phase and mouse use the same conversion logic
            
            def convert_char_array_to_string(value, field_name='field'):
                """Convert a MATLAB char array or string value to Python string.
                
                Handles bytes, numpy arrays (S/U/i/u/f dtype), and other types.
                Uses the same logic for both phase and mouse.
                """
                if isinstance(value, bytes):
                    return value.decode('utf-8').strip('\x00').strip()
                elif isinstance(value, np.ndarray):
                    # Handle numpy arrays - MATLAB char arrays are 2D
                    if value.dtype.kind == 'S':  # String array (bytes)
                        # Convert bytes array to string
                        if value.ndim == 2:
                            # 2D array like [1, N] - flatten and decode
                            return value.flatten().tobytes().decode('utf-8').strip('\x00').strip()
                        else:
                            return value.tobytes().decode('utf-8').strip('\x00').strip()
                    elif value.dtype.kind == 'U':  # Unicode string array
                        # Unicode array - convert to string
                        if value.ndim == 2:
                            return ''.join(value.flatten()).strip()
                        else:
                            return str(value.item()).strip()
                    elif value.dtype.kind in ['i', 'u', 'f']:  # Integer, unsigned int, float
                        # Numeric array - convert to string
                        return str(value.item() if value.size == 1 else value.flat[0]).strip()
                    else:
                        # Other array types - try to convert
                        if value.ndim == 2:
                            # 2D array - try to convert as char array
                            try:
                                return ''.join([chr(int(c)) for c in value.flatten() if c != 0]).strip()
                            except (ValueError, TypeError):
                                return str(value.flat[0]).strip()
                        elif value.size == 1:
                            return str(value.item()).strip()
                        else:
                            return str(value.flat[0]).strip()
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    # Numeric type - convert to string
                    return str(value).strip()
                else:
                    # Already a string or other type - ensure it's a string
                    return str(value).strip()
            
            # Convert phase to string using shared function
            phase = convert_char_array_to_string(phase, 'phase')
            
            # Convert mouse to string using the same function (identical logic as phase)
            mouse = convert_char_array_to_string(mouse, 'mouse')
            
            # Ensure both are non-empty strings
            if not phase:
                logger.warning(f"Empty phase value for dataset {i}, using 'unknown'")
                phase = 'unknown'
            if not mouse:
                logger.warning(f"Empty mouse value for dataset {i}, using 'unknown'")
                mouse = 'unknown'
            
            # Store in table
            # Phase and mouse are stored as Python strings (e.g., 'late', 'early', 'HB059', '72')
            # They preserve alphanumeric values from MATLAB preprocessing
            data_table[i, 0] = dff
            data_table[i, 1] = zscore
            data_table[i, 2] = stim
            data_table[i, 3] = response
            data_table[i, 4] = phase  # String: 'early', 'mid', 'late', etc.
            data_table[i, 5] = mouse  # String: 'HB059', '72', '101', etc.
            
            # Debug: Log first few rows to verify column order
            if i < 3:
                logger.debug(f"Row {i}: phase='{phase}', mouse='{mouse}'")
        
        # Verify phase values are actually phase strings, not mouse IDs
        sample_phases = [str(data_table[i, 4]) for i in range(min(10, n_datasets))]
        sample_mice = [str(data_table[i, 5]) for i in range(min(10, n_datasets))]
        logger.info(f"Sample phases (first 10): {sample_phases}")
        logger.info(f"Sample mice (first 10): {sample_mice}")
        
        # Check if phases look like mouse IDs (numeric strings) and mice look like phase strings
        unique_phases = set(sample_phases)
        unique_mice = set(sample_mice)
        phases_are_numeric = all(p.isdigit() for p in unique_phases if p)
        mice_are_strings = any(m.lower() in ['early', 'mid', 'late'] for m in unique_mice if isinstance(m, str))
        mice_are_numeric = all(m.isdigit() for m in unique_mice if m)
        
        # Check if phase column contains multiple unique numeric values (like mouse IDs)
        # and mouse column contains fewer unique values (like a single mouse ID)
        # This suggests columns might be swapped
        if phases_are_numeric and len(unique_phases) > len(unique_mice) and mice_are_numeric:
            logger.warning(f"⚠️  WARNING: Phase column contains multiple numeric values: {unique_phases}")
            logger.warning(f"⚠️  Mouse column contains fewer numeric values: {unique_mice}")
            logger.warning(f"⚠️  This suggests columns may be swapped (phase should be 'early'/'mid'/'late', not numeric)")
            logger.warning(f"⚠️  Attempting to swap columns 4 and 5 (phase <-> mouse)")
            
            # Swap phase and mouse columns
            for i in range(n_datasets):
                phase_val = data_table[i, 4]
                mouse_val = data_table[i, 5]
                data_table[i, 4] = mouse_val  # Mouse becomes phase
                data_table[i, 5] = phase_val  # Phase becomes mouse
            
            # Re-check after swap
            new_sample_phases = [str(data_table[i, 4]) for i in range(min(10, n_datasets))]
            new_sample_mice = [str(data_table[i, 5]) for i in range(min(10, n_datasets))]
            new_unique_phases = set(new_sample_phases)
            new_unique_mice = set(new_sample_mice)
            
            logger.info(f"✅ Swapped columns. New sample phases: {new_sample_phases}")
            logger.info(f"✅ New sample mice: {new_sample_mice}")
            
            # Verify swap worked
            if any(p.lower() in ['early', 'mid', 'late'] for p in new_unique_phases if isinstance(p, str)):
                logger.info(f"✅ Swap successful: phases now contain phase strings: {new_unique_phases}")
            else:
                logger.warning(f"⚠️  Swap may not have fixed the issue. Phases still: {new_unique_phases}")
        elif phases_are_numeric and mice_are_strings:
            logger.warning(f"⚠️  WARNING: Phase and mouse columns appear to be swapped!")
            logger.warning(f"⚠️  Phase column contains: {unique_phases} (looks like mouse IDs)")
            logger.warning(f"⚠️  Mouse column contains: {unique_mice} (looks like phase strings)")
            logger.warning(f"⚠️  Swapping columns 4 and 5 (phase <-> mouse)")
            
            # Swap phase and mouse columns
            for i in range(n_datasets):
                phase_val = data_table[i, 4]
                mouse_val = data_table[i, 5]
                data_table[i, 4] = mouse_val  # Mouse becomes phase
                data_table[i, 5] = phase_val  # Phase becomes mouse
            
            logger.info(f"✅ Swapped columns. New sample phases: {[str(data_table[i, 4]) for i in range(min(10, n_datasets))]}")
            logger.info(f"✅ New sample mice: {[str(data_table[i, 5]) for i in range(min(10, n_datasets))]}")
        elif phases_are_numeric:
            logger.warning(f"⚠️  WARNING: Phase column contains numeric values: {unique_phases}")
            logger.warning(f"⚠️  Expected phases: 'early', 'mid', 'late'")
            logger.warning(f"⚠️  Mouse column values: {unique_mice}")
            logger.warning(f"⚠️  This may indicate a data issue or column swap that couldn't be auto-detected")
        
        logger.info(f"Loaded {n_datasets} datasets from preprocessed file")
        return data_table
        
    def _load_h5py_table_simple(self, h5_file):
        """Load table data from HDF5 format - keep datasets separate for train/val splitting."""
        logger.info("Analyzing file structure...")
        
        # The file contains datasets like A, B, C... for neural data
        # and smaller datasets like Ao, Ap, Aq... for metadata
        
        # Find neural data arrays (large 3D arrays: time_points x brain_areas x trials)
        neural_data_keys = []
        metadata_keys = {}
        
        # Look specifically in the #refs# group where the data is stored
        if '#refs#' in h5_file:
            refs_group = h5_file['#refs#']
            for key in refs_group.keys():
                dataset = refs_group[key]
                if hasattr(dataset, 'shape'):
                    logger.debug(f"  Checking {key}: shape={dataset.shape}, ndim={len(dataset.shape)}")
                    
                    if len(dataset.shape) == 3:
                        # This looks like neural data (time_points, brain_areas, n_trials)
                        time_points, brain_areas, n_trials = dataset.shape
                        
                        if time_points == 41 and brain_areas == 82:  # Based on inspection: 41 time points, 82 brain areas
                            neural_data_keys.append(key)
                            logger.debug(f"  ✓ Found neural data {key}: {dataset.shape}")
                            
                            # Look for corresponding metadata in the same group
                            base_key = key
                            metadata_keys[base_key] = {}
                            
                            # Common suffixes for metadata
                            suffixes = ['o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
                            for suffix in suffixes:
                                meta_key = base_key + suffix
                                if meta_key in refs_group:
                                    meta_data = refs_group[meta_key]
                                    if hasattr(meta_data, 'shape'):
                                        metadata_keys[base_key][suffix] = meta_data[()]
                        else:
                            logger.debug(f"    Skipped {key}: brain_areas={brain_areas}, time_points={time_points}")
        else:
            logger.error("No #refs# group found in file!")
        
        if not neural_data_keys:
            raise ValueError("No neural data arrays found in file")
        
        logger.info(f"Found {len(neural_data_keys)} neural datasets")
        
        # NEW APPROACH: Keep datasets separate for proper train/val splitting
        # Create table structure with one row per dataset
        n_rows = len(neural_data_keys)
        column_names = ['dff', 'zscore', 'stim', 'response', 'phase', 'mouse']
        n_cols = len(column_names)
        
        data_table = np.empty((n_rows, n_cols), dtype=object)
        
        refs_group = h5_file['#refs#']
        
        for dataset_idx, neural_key in enumerate(neural_data_keys):
            # Load neural data from the refs group
            neural_data = refs_group[neural_key][()]  # Shape: (time_points, brain_areas, trials)
            
            # Original data is (time_points, brain_areas, trials)
            # User wants (trials, time_points, brain_areas)
            neural_data_transposed = neural_data.transpose(2, 0, 1)  # Convert to (trials, time_points, brain_areas)
            
            n_trials = neural_data.shape[2]  # Number of trials in this dataset
            
            # Load metadata if available
            # Expected MATLAB table column order: dff, zscore, stim, response, phase, mouse
            # Map metadata suffixes to columns in order
            # Common suffixes: o, p, q, r, s, t, u, v, w, x, y, z
            # Assuming order: o=dff, p=zscore, q=stim, r=response, s=phase, t=mouse (or similar)
            meta_data = metadata_keys.get(neural_key, {})
            meta_suffixes = sorted(meta_data.keys())  # Sort for consistent ordering
            
            # Expected column order: dff, zscore, stim, response, phase, mouse
            # Map suffixes to column indices (assuming sequential mapping)
            suffix_to_column = {
                'o': 0,  # dff (neural data, already loaded)
                'p': 1,  # zscore (neural data, already loaded)
                'q': 2,  # stim
                'r': 3,  # response
                's': 4,  # phase
                't': 5,  # mouse
                'u': 2,  # fallback stim
                'v': 3,  # fallback response
                'w': 4,  # fallback phase
                'x': 5,  # fallback mouse
            }
            
            stim_data = None
            response_data = None
            phase_data = None
            
            # Try to map suffixes to columns in order
            for suffix in meta_suffixes:
                if suffix not in meta_data:
                    continue
                
                meta_val = meta_data[suffix]
                if meta_val is None:
                    continue
                
                # Flatten and convert to array
                if hasattr(meta_val, 'flatten'):
                    flat_val = meta_val.flatten()
                else:
                    flat_val = np.array(meta_val).flatten()
                
                # Skip if wrong shape (should match n_trials)
                if len(flat_val) != n_trials:
                    logger.debug(f"  Metadata {neural_key}{suffix}: shape mismatch ({len(flat_val)} vs {n_trials}), skipping")
                    continue
                
                # Map suffix to column based on expected order
                if suffix in ['q', 'u'] and stim_data is None:
                    # This should be stim column (3rd column, index 2)
                    try:
                        stim_data = flat_val.astype(int)
                        unique_vals = sorted(np.unique(stim_data))
                        logger.debug(f"  Mapped {neural_key}{suffix} -> stim: unique values {unique_vals}")
                    except (ValueError, TypeError):
                        logger.debug(f"  Could not convert {neural_key}{suffix} to int for stim")
                
                elif suffix in ['r', 'v'] and response_data is None:
                    # This should be response column (4th column, index 3)
                    try:
                        response_data = flat_val.astype(int)
                        unique_vals = sorted(np.unique(response_data))
                        logger.debug(f"  Mapped {neural_key}{suffix} -> response: unique values {unique_vals}")
                    except (ValueError, TypeError):
                        logger.debug(f"  Could not convert {neural_key}{suffix} to int for response")
                
                elif suffix in ['s', 'w'] and phase_data is None:
                    # This should be phase column (5th column, index 4)
                    try:
                        # Phase might be strings or integers
                        if flat_val.dtype.kind in ['U', 'S', 'O']:  # String types
                            phase_data = [str(p) for p in flat_val]
                        else:
                            # Convert integers to phase names
                            phase_map = {0: 'early', 1: 'mid', 2: 'late'}
                            phase_data = [phase_map.get(int(p), 'early') for p in flat_val]
                        unique_vals = sorted(set(phase_data))
                        logger.debug(f"  Mapped {neural_key}{suffix} -> phase: unique values {unique_vals}")
                    except (ValueError, TypeError):
                        logger.debug(f"  Could not convert {neural_key}{suffix} to phase")
            
            # Fallback: if we couldn't map by suffix order, try identifying by value ranges
            if stim_data is None or response_data is None:
                logger.debug(f"  Could not map all columns by suffix order for {neural_key}, trying value-based identification...")
                for suffix, meta_val in meta_data.items():
                    if meta_val is None:
                        continue
                    
                    if hasattr(meta_val, 'flatten'):
                        flat_val = meta_val.flatten()
                    else:
                        flat_val = np.array(meta_val).flatten()
                    
                    if len(flat_val) != n_trials:
                        continue
                    
                    # Check if values are small integers (likely stim/response)
                    try:
                        unique_vals = np.unique(flat_val)
                        unique_ints = unique_vals.astype(int)
                        if np.all((unique_ints >= 0) & (unique_ints <= 10)):
                            if stim_data is None:
                                stim_data = flat_val.astype(int)
                                logger.debug(f"  Identified stim from {neural_key}{suffix}: unique values {sorted(unique_ints)}")
                            elif response_data is None:
                                response_data = flat_val.astype(int)
                                logger.debug(f"  Identified response from {neural_key}{suffix}: unique values {sorted(unique_ints)}")
                    except (ValueError, TypeError):
                        continue
            
            # Final fallback to defaults if not found
            # Note: These warnings are expected when using the fallback loader
            # Prefer using preprocessed data (created by preprocess_matlab_table.m) to avoid this
            if stim_data is None:
                logger.debug(f"Could not identify stim data for {neural_key}, using defaults (consider using preprocessed data)")
                stim_data = np.ones(n_trials, dtype=int)
            if response_data is None:
                logger.debug(f"Could not identify response data for {neural_key}, using defaults (consider using preprocessed data)")
                response_data = np.zeros(n_trials, dtype=int)
            if phase_data is None:
                # Phase: create dummy data alternating between early/mid/late across datasets
                if dataset_idx % 3 == 0:
                    phase_data = ["early"] * n_trials
                elif dataset_idx % 3 == 1:
                    phase_data = ["mid"] * n_trials
                else:
                    phase_data = ["late"] * n_trials
                
                # Ensure correct shapes and types
                if hasattr(stim_data, 'flatten'):
                    stim_data = stim_data.flatten()
                if hasattr(response_data, 'flatten'):
                    response_data = response_data.flatten()
                
                # Truncate/pad to match number of trials if needed
                stim_data = stim_data[:n_trials] if len(stim_data) >= n_trials else np.pad(stim_data, (0, n_trials - len(stim_data)), constant_values=1)
                response_data = response_data[:n_trials] if len(response_data) >= n_trials else np.pad(response_data, (0, n_trials - len(response_data)), constant_values=0)
                
            else:
                # Create dummy metadata
                stim_data = np.ones(n_trials, dtype=int)  # All stim = 1
                response_data = np.ones(n_trials, dtype=int)  # All response = 1 for valid trials
                
                # Phase: cycle through early/mid/late across datasets
                if dataset_idx % 3 == 0:
                    phase_data = ["early"] * n_trials
                elif dataset_idx % 3 == 1:
                    phase_data = ["mid"] * n_trials
                else:
                    phase_data = ["late"] * n_trials
                
                logger.warning(f"Created dummy metadata for {neural_key}")
            
            # Store data in the table row
            data_table[dataset_idx, 0] = neural_data_transposed  # dff data
            data_table[dataset_idx, 1] = neural_data_transposed  # zscore data (same for now)
            data_table[dataset_idx, 2] = stim_data
            data_table[dataset_idx, 3] = response_data
            data_table[dataset_idx, 4] = np.array(phase_data)
            data_table[dataset_idx, 5] = neural_key  # Mouse ID (dataset key)
        
        logger.info(f"Created data table with {n_rows} datasets")
        logger.info(f"Sample dataset shape: {data_table[0, 0].shape}")
        
        # Log dataset statistics
        total_trials = sum(data_table[i, 0].shape[0] for i in range(n_rows))
        phase_counts = {}
        for i in range(n_rows):
            phases = data_table[i, 4]
            for phase in np.unique(phases):
                phase_counts[phase] = phase_counts.get(phase, 0) + np.sum(phases == phase)
        
        logger.info(f"Dataset statistics:")
        logger.info(f"  Total datasets: {n_rows}")
        logger.info(f"  Total trials: {total_trials}")
        logger.info(f"  Time points: {data_table[0, 0].shape[1]}")
        logger.info(f"  Brain areas: {data_table[0, 0].shape[2]}")
        logger.info(f"  Phase distribution: {phase_counts}")
        
        return data_table
        
    def get_column_names(self) -> List[str]:
        """Get the column names from the data table."""
        return ['dff', 'zscore', 'stim', 'response', 'phase', 'mouse']
    
    def extract_column_data(self, column_name: str) -> List[np.ndarray]:
        """
        Extract data from a specific column.
        
        Args:
            column_name: Name of the column to extract
            
        Returns:
            List of numpy arrays, one for each row in the table
        """
        column_names = self.get_column_names()
        if column_name not in column_names:
            raise ValueError(f"Column {column_name} not found. Available: {column_names}")
            
        column_idx = column_names.index(column_name)
        
        # Extract data from each row for the specified column
        column_data = []
        for row_idx in range(self.data_table.shape[0]):
            data = self.data_table[row_idx, column_idx]
            # Handle different data formats
            if hasattr(data, 'shape') and data.shape == ():
                # Scalar array
                data = data.item()
            elif hasattr(data, 'shape') and len(data.shape) == 1 and data.shape[0] == 1:
                # Single-element 1D array
                data = data.item()
            # For multi-dimensional arrays, keep the original shape
            # Don't flatten - this preserves 3D neural data structure
            column_data.append(data)
            
        return column_data
    
    def create_cross_validation_splits(self, n_splits: int = 5) -> List[Tuple[List[int], List[int]]]:
        """
        Create stratified cross-validation splits ensuring:
        1. No data leakage between mice (mouse-based grouping)
        2. Both phases represented in training and validation sets
        
        NOTE: Each row is a SESSION (not a dataset). Sessions contain multiple trials.
        All sessions from the same mouse must go to the same split (train/val/test).
        Splits sessions (rows) instead of trials, with 80%/20% split.
        
        Args:
            n_splits: Number of cross-validation folds (ignored for 80/20 split)
            
        Returns:
            List with single (train_indices, val_indices) tuple for 80/20 split
            NOTE: These are session-level indices (row indices)
        """
        n_sessions = self.data_table.shape[0]
        logger.info(f"Creating 80/20 session split for {n_sessions} total sessions")
        
        # Get mouse IDs (one per session)
        # Each row is a session, and we need to ensure all sessions from the same mouse
        # go to the same split (train/val/test)
        mouse_ids = []
        for row_idx in range(n_sessions):
            mouse_id_raw = self.data_table[row_idx, 5]  # mouse column
            
            # Normalize mouse ID: handle various formats
            # mouse_id_raw should already be a string from preprocessing, but handle edge cases
            if isinstance(mouse_id_raw, str):
                # Already a string - use as-is (don't iterate over it!)
                mouse_id = mouse_id_raw.strip()
            elif isinstance(mouse_id_raw, bytes):
                mouse_id = mouse_id_raw.decode('utf-8').strip('\x00').strip()
            elif isinstance(mouse_id_raw, np.ndarray):
                if mouse_id_raw.dtype.kind == 'S':  # String array (bytes)
                    mouse_id = mouse_id_raw.tobytes().decode('utf-8').strip('\x00').strip()
                elif mouse_id_raw.dtype.kind == 'U':  # Unicode string array
                    if mouse_id_raw.ndim == 0 or mouse_id_raw.size == 1:
                        mouse_id = str(mouse_id_raw.item()).strip()
                    else:
                        # Array of characters - join them
                        mouse_id = ''.join([str(c) for c in mouse_id_raw.flatten()]).strip()
                elif mouse_id_raw.size == 1:
                    # Single element array - convert to string
                    mouse_id = str(mouse_id_raw.item()).strip()
                else:
                    # Array - try to convert as char array (numeric codes) or take first element
                    try:
                        # Try as char array (numeric codes)
                        chars = [chr(int(c)) for c in mouse_id_raw.flatten() if int(c) != 0 and 32 <= int(c) <= 126]
                        mouse_id = ''.join(chars).strip() if chars else str(mouse_id_raw.flat[0]).strip()
                    except (ValueError, TypeError, OverflowError):
                        # Fallback: take first element
                        mouse_id = str(mouse_id_raw.flat[0]).strip()
            else:
                # Other type - convert to string (but don't iterate if it's already a string-like object)
                mouse_id = str(mouse_id_raw).strip()
            
            mouse_ids.append(mouse_id)
        
        unique_mice = list(set(mouse_ids))
        logger.info(f"Found {len(unique_mice)} unique mice across {n_sessions} sessions")
        
        # Debug: Show sample mouse IDs and their distribution
        if len(unique_mice) <= 10:
            logger.info(f"Unique mouse IDs: {unique_mice}")
        else:
            logger.info(f"Sample mouse IDs (first 10): {unique_mice[:10]}")
        
        # Show distribution of sessions per mouse
        mouse_counts = Counter(mouse_ids)
        logger.info(f"Sessions per mouse distribution: {dict(mouse_counts.most_common(10))}")
        
        # Group sessions by mouse ID to ensure no mouse overlap
        mouse_to_sessions = {}
        for row_idx, mouse_id in enumerate(mouse_ids):
            if mouse_id not in mouse_to_sessions:
                mouse_to_sessions[mouse_id] = []
            mouse_to_sessions[mouse_id].append(row_idx)
        
        logger.info(f"Mice with multiple sessions: {sum(1 for v in mouse_to_sessions.values() if len(v) > 1)}")
        
        # Get phase information for each session
        session_phases = []
        for row_idx in range(n_sessions):
            phases = self.data_table[row_idx, 4]  # phase column
            # Phase is a single string per row
            if isinstance(phases, str):
                session_phases.append(phases)
            elif hasattr(phases, '__len__') and len(phases) == 1:
                session_phases.append(str(phases[0] if hasattr(phases, '__getitem__') else phases))
            else:
                # Fallback: get first phase or default
                session_phases.append(str(phases) if not hasattr(phases, '__iter__') else str(phases[0]))
        
        # Count sessions per phase
        phase_counts = Counter(session_phases)
        logger.info(f"Session phase distribution: {dict(phase_counts)}")
        
        # Get dominant phase for each mouse (for stratification)
        mouse_phases = {}
        for mouse_id, session_indices_for_mouse in mouse_to_sessions.items():
            phases_for_mouse = [session_phases[idx] for idx in session_indices_for_mouse]
            # Use most common phase for this mouse
            phase_counts_for_mouse = Counter(phases_for_mouse)
            dominant_phase = phase_counts_for_mouse.most_common(1)[0][0]
            mouse_phases[mouse_id] = dominant_phase
        
        # Split mice (not sessions) to ensure no mouse overlap
        from sklearn.model_selection import train_test_split
        
        mouse_list = list(mouse_to_sessions.keys())
        mouse_phase_list = [mouse_phases[mouse_id] for mouse_id in mouse_list]
        
        # Handle edge case: if only 1 mouse, put all sessions in training
        if len(mouse_list) == 1:
            logger.warning(f"Only 1 unique mouse found. All sessions will be used for training (no validation split possible).")
            train_mice = mouse_list
            val_mice = []
        elif len(mouse_list) == 2:
            # With 2 mice, use 1 for train, 1 for val
            logger.info(f"Only 2 unique mice found. Using 1 mouse for training, 1 for validation.")
            train_mice = [mouse_list[0]]
            val_mice = [mouse_list[1]]
        else:
            # Normal case: 3+ mice, can do proper split
            try:
                # Try stratified split by mouse phase to maintain phase balance
                train_mice, val_mice = train_test_split(
                    mouse_list,
                    test_size=0.2,
                    stratify=mouse_phase_list,
                    random_state=42  # For reproducibility
                )
            except ValueError as e:
                logger.warning(f"Could not perform stratified split by mouse phase: {e}")
                logger.warning("Falling back to random mouse split")
                # Fall back to random split if stratification fails
                try:
                    train_mice, val_mice = train_test_split(
                        mouse_list,
                        test_size=0.2,
                        random_state=42
                    )
                except ValueError as e2:
                    # If still fails (e.g., test_size too large), use manual split
                    logger.warning(f"train_test_split failed: {e2}. Using manual split.")
                    n_val_mice = max(1, int(len(mouse_list) * 0.2))
                    val_mice = mouse_list[:n_val_mice]
                    train_mice = mouse_list[n_val_mice:]
        
        # Collect all session indices for train and val mice
        # All sessions from a given mouse go to the same split
        train_indices = []
        val_indices = []
        
        for mouse_id in train_mice:
            train_indices.extend(mouse_to_sessions[mouse_id])
        
        for mouse_id in val_mice:
            val_indices.extend(mouse_to_sessions[mouse_id])
        
        train_indices = sorted(train_indices)
        val_indices = sorted(val_indices)
        
        logger.info(f"Split sizes: {len(train_indices)} train sessions ({len(train_mice)} mice), {len(val_indices)} val sessions ({len(val_mice)} mice)")
        
        # Log split statistics
        train_phases = [session_phases[i] for i in train_indices]
        train_phase_counts = Counter(train_phases)
        
        logger.info(f"Train split: {len(train_indices)} sessions")
        logger.info(f"  Phase distribution: {dict(train_phase_counts)}")
        
        if len(val_indices) > 0:
            val_phases = [session_phases[i] for i in val_indices]
            val_phase_counts = Counter(val_phases)
            logger.info(f"Val split: {len(val_indices)} sessions")
            logger.info(f"  Phase distribution: {dict(val_phase_counts)}")
        else:
            logger.warning(f"Val split: 0 sessions (no validation data)")
        
        # Check for missing phases
        available_phases = list(phase_counts.keys())
        train_missing_phases = [p for p in available_phases if train_phase_counts.get(p, 0) == 0]
        
        if train_missing_phases:
            logger.warning(f"⚠️  Training missing phases: {train_missing_phases}")
        
        if len(val_indices) > 0:
            val_phase_counts = Counter([session_phases[i] for i in val_indices])
            val_missing_phases = [p for p in available_phases if val_phase_counts.get(p, 0) == 0]
            if val_missing_phases:
                logger.warning(f"⚠️  Validation missing phases: {val_missing_phases}")
        
        # Verify no mouse overlap (only if we have validation data)
        if len(val_indices) > 0:
            train_mice = [mouse_ids[i] for i in train_indices]
            val_mice = [mouse_ids[i] for i in val_indices]
            
            overlap = set(train_mice) & set(val_mice)
            if overlap:
                logger.error(f"❌ Mouse overlap detected: {overlap}")
                raise ValueError(f"Mouse overlap between train and validation: {overlap}")
            else:
                logger.info(f"✅ No mouse overlap - {len(set(train_mice))} train mice, {len(set(val_mice))} val mice")
        else:
            logger.warning(f"⚠️  No validation data (all datasets in training set)")
            logger.info(f"✅ All {len(set([mouse_ids[i] for i in train_indices]))} mice in training set")
        
        # Return as list with single split for compatibility
        return [(train_indices, val_indices)]
    
    def _rebalance_phases(
        self, 
        train_idx: np.ndarray, 
        val_idx: np.ndarray, 
        phases: List[str], 
        phase_mouse_groups: Dict[str, List], 
        mouse_ids: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Attempt to rebalance phases between train and validation sets.
        
        Args:
            train_idx: Current training indices
            val_idx: Current validation indices
            phases: Phase for each row
            phase_mouse_groups: Mice grouped by phase
            mouse_ids: Mouse ID for each row
            
        Returns:
            Rebalanced (train_idx, val_idx)
        """
        train_set = set(train_idx)
        val_set = set(val_idx)
        
        train_phases = Counter(phases[i] for i in train_idx)
        val_phases = Counter(phases[i] for i in val_idx)
        
        available_phases = list(phase_mouse_groups.keys())
        
        # Find missing phases
        train_missing = [p for p in available_phases if train_phases.get(p, 0) == 0]
        val_missing = [p for p in available_phases if val_phases.get(p, 0) == 0]
        
        # Try to move mice to balance phases
        for missing_phase in train_missing:
            # Find mice of this phase in validation set
            candidates = [i for i in val_idx if phases[i] == missing_phase]
            if candidates:
                # Move one mouse (and all its rows) from val to train
                mouse_to_move = mouse_ids[candidates[0]]
                rows_to_move = [i for i in val_idx if mouse_ids[i] == mouse_to_move]
                
                for row in rows_to_move:
                    train_set.add(row)
                    val_set.discard(row)
                
                logger.info(f"    Moved mouse {mouse_to_move} ({len(rows_to_move)} rows) from val to train for phase {missing_phase}")
        
        for missing_phase in val_missing:
            # Find mice of this phase in training set
            candidates = [i for i in train_idx if phases[i] == missing_phase]
            if candidates:
                # Move one mouse (and all its rows) from train to val
                mouse_to_move = mouse_ids[candidates[0]]
                rows_to_move = [i for i in train_idx if mouse_ids[i] == mouse_to_move]
                
                for row in rows_to_move:
                    val_set.add(row)
                    train_set.discard(row)
                
                logger.info(f"    Moved mouse {mouse_to_move} ({len(rows_to_move)} rows) from train to val for phase {missing_phase}")
        
        return np.array(sorted(train_set)), np.array(sorted(val_set))


class TaskDefinition:
    """
    Define classification tasks based on stimulus and response conditions.
    """
    
    def __init__(
        self, 
        stim_values: List[int],
        response_values: List[int], 
        phases: List[str],
        task_name: str = "custom_task"
    ):
        """
        Initialize task definition.
        
        Args:
            stim_values: List of stimulus values to include
            response_values: List of response values to include  
            phases: List of phase values to include ('early', 'mid', 'late')
            task_name: Name for this task
        """
        self.stim_values = stim_values
        self.response_values = response_values
        # Normalize phases: strip whitespace and convert to lowercase for consistent matching
        # Also handle integer phases: 0='early', 1='mid', 2='late'
        # Note: string '1', integer 1, and string 'int' are all treated as equivalent (all map to 'mid')
        phase_map_int = {0: 'early', 1: 'mid', 2: 'late'}
        normalized_phases = []
        for p in phases:
            # Check if it's a string representation of an integer (e.g., "1", "0", "2")
            if isinstance(p, str) and p.strip().isdigit():
                # Convert string "1" to integer 1, then map to string
                p_int = int(p.strip())
                normalized_phases.append(phase_map_int.get(p_int, str(p_int).lower()))
            elif isinstance(p, str) and p.strip().lower() == 'int':
                # String 'int' maps to 'mid' (since integer 1 maps to 'mid')
                normalized_phases.append('mid')
            elif isinstance(p, (int, np.integer)):
                # Integer phase - map to string
                normalized_phases.append(phase_map_int.get(int(p), str(p).strip().lower()))
            else:
                # String phase - normalize
                normalized_phases.append(str(p).strip().lower())
        self.phases = normalized_phases
        self.task_name = task_name
        
    def filter_trials(
        self, 
        dataset: WidefieldDataset, 
        dataset_indices: List[int],
        data_type: str = 'dff',
        return_target_column: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Optional[np.ndarray]]:
        """
        Filter trials based on task definition and combine datasets.
        
        NOTE: Now works with dataset-level indices and combines the selected datasets
        into single arrays for training/validation.
        
        Args:
            dataset: WidefieldDataset instance
            dataset_indices: Indices of datasets (rows) to consider
            data_type: Type of data to extract ('dff' or 'zscore')
            return_target_column: If 'stim' or 'response', also return raw values for regression
            
        Returns:
            Tuple of (combined_neural_data, combined_labels, combined_trial_indices, mouse_ids)
            or 5-tuple with target_values as last element when return_target_column is set
        """
        combined_neural_data = []
        combined_labels = []
        combined_trial_indices = []
        combined_mouse_ids = []
        combined_target_values = [] if return_target_column in ('stim', 'response') else None
        
        # Track NaN filtering statistics
        total_candidate_trials = 0
        nan_filtered_trials = 0
        
        for dataset_idx in dataset_indices:
            if dataset_idx >= dataset.data_table.shape[0]:
                logger.warning(f"Dataset index {dataset_idx} out of bounds, skipping")
                continue
                
            # Extract data from this dataset
            neural_data = dataset.data_table[dataset_idx, 0 if data_type == 'dff' else 1]  # dff=0, zscore=1
            stim_data = dataset.data_table[dataset_idx, 2]  # stim column
            response_data = dataset.data_table[dataset_idx, 3]  # response column
            phase_data = dataset.data_table[dataset_idx, 4]  # phase column
            mouse_id = dataset.data_table[dataset_idx, 5]  # mouse column
            
            # Validate stim and response data
            if stim_data is None or (hasattr(stim_data, '__len__') and len(stim_data) == 0):
                logger.warning(f"Dataset {dataset_idx}: stim_data is empty or None, skipping")
                continue
            if response_data is None or (hasattr(response_data, '__len__') and len(response_data) == 0):
                logger.warning(f"Dataset {dataset_idx}: response_data is empty or None, skipping")
                continue
            
            # Handle data format: neural_data can be in different formats
            # Confirmed structure from .mat file: [225 82 41] = (trials, brain_areas, timepoints)
            # But some datasets might be: (brain_areas, timepoints, trials) = (82, 41, 225)
            # Expected format for processing: (trials, timepoints, brain_areas) = (N, timepoints, brain_areas)
            # CDKL5 data: (trials, timepoints=30, brain_areas=56) - PREPROCESSED FORMAT
            if neural_data.ndim == 3:
                original_shape = neural_data.shape
                n1, n2, n3 = neural_data.shape
                
                # Log shape for debugging (first few datasets)
                if dataset_idx < 3:
                    logger.info(f"Dataset {dataset_idx}: Raw neural_data shape from MATLAB: {original_shape}")
                    logger.info(f"Dataset {dataset_idx}: Expected CDKL5: (trials, 30, 56)")
                    logger.info(f"Dataset {dataset_idx}: Actual dimensions: dim0={n1}, dim1={n2}, dim2={n3}")
                
                # Heuristic: determine format based on typical values
                # Typical: trials < 1000, timepoints ~30-50, brain_areas ~50-2000
                # If first dim is smallest and reasonable for trials, likely (trials, timepoints, brain_areas) - already correct
                # If first dim is large (>100), likely (brain_areas, timepoints, trials) or (timepoints, brain_areas, trials)
                
                # Check for CDKL5 data format: MATLAB saves as (brain_areas=56, timepoints=30, trials)
                # Need to transpose to (trials, timepoints=30, brain_areas=56)
                if n1 == 56 and n2 == 30:
                    # CDKL5 format saved by MATLAB: (brain_areas=56, timepoints=30, trials) -> transpose to (trials, 30, 56)
                    neural_data = np.transpose(neural_data, (2, 1, 0))
                    logger.info(f"CDKL5 data detected: Transposed from {original_shape} (brain_areas=56, timepoints=30, trials) to {neural_data.shape} (trials, timepoints=30, brain_areas=56)")
                # Check for CDKL5 data format: (trials, 30, 56) - should NOT transpose
                # CDKL5 preprocessing creates (trials, 30, 56) which is already correct
                elif n2 == 30 and n3 == 56:
                    # CDKL5 format: (trials, 30, 56) - already correct, do NOT transpose
                    logger.info(f"CDKL5 data detected: {neural_data.shape} (trials, timepoints=30, brain_areas=56) - keeping as-is")
                # Check if already in correct format (trials, timepoints, brain_areas)
                # CDKL5: (trials, 30, brain_areas) where brain_areas is 56
                elif n1 < 1000 and n2 < 100 and n3 > 50:
                    # Likely (trials, timepoints, brain_areas) - already correct
                    logger.debug(f"Neural data appears to be in correct format: {neural_data.shape} (trials, timepoints, brain_areas)")
                # Check if shape is (trials, brain_areas, timepoints) - need to transpose
                # BUT exclude CDKL5 format (trials, 56, 30) which should NOT be transposed if n2==56 and n3==30
                elif n1 < 1000 and n2 > 50 and n3 < 100 and not (n2 == 56 and n3 == 30):
                    # Format: (trials, brain_areas, timepoints) -> transpose to (trials, timepoints, brain_areas)
                    # But NOT if it's CDKL5 format (trials, 56, 30) which is actually wrong orientation
                    neural_data = np.transpose(neural_data, (0, 2, 1))
                    logger.info(f"Transposed neural_data from {original_shape} (trials, brain_areas, timepoints) to {neural_data.shape} (trials, timepoints, brain_areas)")
                # Check if shape is (brain_areas, timepoints, trials) - general case
                elif n1 > 50 and n2 < 100 and n3 < 1000:
                    # Format: (brain_areas, timepoints, trials) -> transpose to (trials, timepoints, brain_areas)
                    neural_data = np.transpose(neural_data, (2, 1, 0))
                    logger.info(f"Transposed neural_data from {original_shape} (brain_areas, timepoints, trials) to {neural_data.shape} (trials, timepoints, brain_areas)")
                # Check if shape is (timepoints, brain_areas, trials)
                elif n1 < 100 and n2 > 50 and n3 < 1000:
                    # Format: (timepoints, brain_areas, trials) -> transpose to (trials, timepoints, brain_areas)
                    neural_data = np.transpose(neural_data, (2, 0, 1))
                    logger.info(f"Transposed neural_data from {original_shape} (timepoints, brain_areas, trials) to {neural_data.shape} (trials, timepoints, brain_areas)")
                # Legacy check for exact 82 brain areas, 41 timepoints (widefield data)
                elif n2 == 82 and n3 == 41:
                    # Format: (trials, brain_areas, timepoints) -> transpose to (trials, timepoints, brain_areas)
                    neural_data = np.transpose(neural_data, (0, 2, 1))
                    logger.info(f"Transposed neural_data from {original_shape} (trials, brain_areas=82, timepoints=41) to {neural_data.shape} (trials, timepoints, brain_areas)")
                elif n1 == 82 and n2 == 41:
                    # Format: (brain_areas, timepoints, trials) -> transpose to (trials, timepoints, brain_areas)
                    neural_data = np.transpose(neural_data, (2, 1, 0))
                    logger.info(f"Transposed neural_data from {original_shape} (brain_areas=82, timepoints=41, trials) to {neural_data.shape} (trials, timepoints, brain_areas)")
                elif n1 == 41 and n2 == 82:
                    # Format: (timepoints, brain_areas, trials) -> transpose to (trials, timepoints, brain_areas)
                    neural_data = np.transpose(neural_data, (2, 0, 1))
                    logger.info(f"Transposed neural_data from {original_shape} (timepoints=41, brain_areas=82, trials) to {neural_data.shape} (trials, timepoints, brain_areas)")
                else:
                    # Assume it's already in (trials, timepoints, brain_areas) format
                    logger.info(f"Neural data shape: {neural_data.shape}, assuming format (trials, timepoints, brain_areas)")
            
            n_trials = neural_data.shape[0]
            
            # Handle phase_data: it might be a single string, integer, or an array
            # If it's a single string/scalar, repeat it for all trials
            # Normalize phase strings: strip whitespace and convert to lowercase for consistent matching
            # Also handle integer phases: 0='early', 1='mid', 2='late'
            phase_map_int = {0: 'early', 1: 'mid', 2: 'late'}
            
            def normalize_phase(p):
                """Normalize a phase value (string or int) to lowercase string.
                
                Maps:
                - Integer 1 -> 'mid'
                - String 'mid' -> 'mid'
                - Integer 0 -> 'early'
                - Integer 2 -> 'late'
                - Other values pass through as normalized strings
                """
                # Check if it's a string representation of an integer (e.g., "1", "0", "2")
                if isinstance(p, str) and p.strip().isdigit():
                    # Convert string "1" to integer 1, then map to string
                    p_int = int(p.strip())
                    return phase_map_int.get(p_int, str(p_int).lower())
                elif isinstance(p, str) and p.strip().lower() == 'int':
                    # String 'int' maps to 'mid' (since integer 1 maps to 'mid')
                    return 'mid'
                elif isinstance(p, (int, np.integer)):
                    # Integer phase - map known integers to strings, others pass through
                    return phase_map_int.get(int(p), str(p).strip().lower())
                else:
                    # String phase - normalize to lowercase
                    # 'mid', 1, '1', and 'int' are all equivalent
                    return str(p).strip().lower()
            
            if isinstance(phase_data, str) or isinstance(phase_data, (int, np.integer)) or (hasattr(phase_data, '__len__') and len(phase_data) == 1):
                # Single phase value - repeat for all trials
                if isinstance(phase_data, str):
                    phase_str = normalize_phase(phase_data)
                elif isinstance(phase_data, (int, np.integer)):
                    phase_str = normalize_phase(phase_data)
                else:
                    phase_str = normalize_phase(phase_data[0] if hasattr(phase_data, '__getitem__') else phase_data)
                phase_array = np.array([phase_str] * n_trials)
            elif hasattr(phase_data, '__len__') and len(phase_data) == n_trials:
                # Already an array with correct length
                phase_array = np.array([normalize_phase(p) for p in phase_data])
            else:
                # Fallback: try to use as-is or repeat first element
                try:
                    phase_array = np.array([normalize_phase(p) for p in phase_data])
                except (TypeError, IndexError):
                    phase_str = normalize_phase(phase_data[0] if hasattr(phase_data, '__getitem__') else phase_data)
                    phase_array = np.array([phase_str] * n_trials)
            
            dataset_valid_trials = []
            dataset_labels = []
            dataset_neural_data = []
            
            for trial_idx in range(n_trials):
                try:
                    # Handle different stim/response data formats
                    if hasattr(stim_data, '__getitem__'):
                        stim_val = int(stim_data[trial_idx])
                    else:
                        stim_val = int(stim_data)  # Scalar value
                    
                    # Handle response data - might be missing or all ones (dummy)
                    if response_data is not None:
                        if hasattr(response_data, '__getitem__'):
                            response_val = int(response_data[trial_idx])
                        else:
                            response_val = int(response_data)  # Scalar value
                    else:
                        # No response data - use dummy value that matches all
                        response_val = 1
                except (IndexError, TypeError, ValueError) as e:
                    logger.warning(f"Dataset {dataset_idx}, trial {trial_idx}: Error reading stim/response data: {e}, stim_data type: {type(stim_data)}, response_data type: {type(response_data)}")
                    continue
                
                trial_phase_str = phase_array[trial_idx]
                
                # Include trials that match the task definition criteria
                # If response_values contains only dummy value [1] and response_data is all ones, accept all
                stim_match = stim_val in self.stim_values
                response_match = response_val in self.response_values
                
                if stim_match and response_match:
                    total_candidate_trials += 1
                    
                    # Check if this trial has NaN values
                    # Extract trial data - neural_data is now (trials, timepoints, brain_areas)
                    # After transposition, each trial is (timepoints, brain_areas)
                    trial_neural_data = neural_data[trial_idx, :, :]  # Shape: (timepoints, brain_areas)
                    
                    # Verify shape is (timepoints, brain_areas)
                    if trial_neural_data.ndim != 2:
                        logger.warning(f"Trial {trial_idx} in dataset {dataset_idx} has incorrect dimensions: {trial_neural_data.ndim}, expected 2, shape: {trial_neural_data.shape}")
                        continue
                    
                    # Check for NaN values - be more lenient for CDKL5 data
                    # Only filter if ALL values are NaN, or if more than 50% are NaN
                    nan_count = np.sum(np.isnan(trial_neural_data))
                    nan_percentage = 100.0 * nan_count / trial_neural_data.size if trial_neural_data.size > 0 else 0
                    
                    if nan_count == trial_neural_data.size:
                        # All values are NaN - skip this trial
                        nan_filtered_trials += 1
                        if nan_filtered_trials <= 5:
                            logger.warning(f"Dataset {dataset_idx}, trial {trial_idx}: All values are NaN, skipping")
                        continue
                    elif nan_percentage > 50.0:
                        # More than 50% NaN - skip this trial
                        nan_filtered_trials += 1
                        if nan_filtered_trials <= 5:
                            logger.warning(f"Dataset {dataset_idx}, trial {trial_idx}: Too many NaN values ({nan_percentage:.1f}%), skipping")
                        continue
                    elif nan_count > 0:
                        # Some NaN values - fill them with 0 and continue
                        trial_neural_data = np.nan_to_num(trial_neural_data, nan=0.0)
                        if nan_filtered_trials < 3:  # Log first few
                            logger.debug(f"Dataset {dataset_idx}, trial {trial_idx}: Filled {nan_count} NaN values ({nan_percentage:.1f}%) with 0")
                    
                    # Check for Inf values - replace with finite values
                    inf_count = np.sum(np.isinf(trial_neural_data))
                    if inf_count > 0:
                        trial_neural_data = np.nan_to_num(trial_neural_data, nan=0.0, posinf=1e6, neginf=-1e6)
                        if nan_filtered_trials < 3:
                            logger.debug(f"Dataset {dataset_idx}, trial {trial_idx}: Replaced {inf_count} Inf values")
                    
                    # Skip trials with phases not in our target list
                    # Special handling: if phases=['all'], accept any phase
                    if 'all' in self.phases:
                        # Use phase='all' as label 0 (for binary classification, this will be overridden by genotype)
                        label = 0
                    elif trial_phase_str not in self.phases:
                        logger.debug(f"Trial {trial_idx} in dataset {dataset_idx}: Phase '{trial_phase_str}' not in target phases {self.phases}")
                        continue
                    else:
                        # Generate label based on trial-specific phase using dynamic mapping
                        try:
                            label = self.phases.index(trial_phase_str)
                        except ValueError:
                            logger.warning(f"Phase '{trial_phase_str}' not found in target phases {self.phases} for trial {trial_idx} in dataset {dataset_idx}, skipping")
                            continue
                    
                    target_val = float(stim_val) if return_target_column == 'stim' else (
                        float(response_val) if return_target_column == 'response' else None)
                    dataset_valid_trials.append(trial_idx)
                    dataset_labels.append(label)
                    dataset_neural_data.append(trial_neural_data)
                    if combined_target_values is not None and target_val is not None:
                        combined_target_values.append(target_val)
            
            # Add this dataset's valid trials to the combined data
            if dataset_valid_trials:
                combined_neural_data.extend(dataset_neural_data)
                combined_labels.extend(dataset_labels)
                # Create global trial indices (dataset_idx, trial_idx pairs)
                for trial_idx in dataset_valid_trials:
                    combined_trial_indices.append((dataset_idx, trial_idx))
                # Add mouse ID for each trial
                combined_mouse_ids.extend([str(mouse_id)] * len(dataset_valid_trials))
        
        if not combined_neural_data:
            # Collect available values for better error message
            available_stim_values = set()
            available_response_values = set()
            available_phases = set()
            
            try:
                for dataset_idx in dataset_indices:
                    if dataset_idx >= dataset.data_table.shape[0]:
                        continue
                    stim_data = dataset.data_table[dataset_idx, 2]
                    response_data = dataset.data_table[dataset_idx, 3]
                    phase_data = dataset.data_table[dataset_idx, 4]
                    
                    if hasattr(stim_data, '__iter__') and not isinstance(stim_data, str):
                        try:
                            available_stim_values.update(np.unique(stim_data).astype(int))
                        except Exception:
                            pass
                    if hasattr(response_data, '__iter__') and not isinstance(response_data, str):
                        try:
                            available_response_values.update(np.unique(response_data).astype(int))
                        except Exception:
                            pass
                    # Handle phase_data: it might be a single string, integer, or an array
                    # Normalize phases: strip whitespace and convert to lowercase
                    # Also handle integer phases: 0='early', 1='mid', 2='late'
                    phase_map_int = {0: 'early', 1: 'mid', 2: 'late'}
                    
                    def normalize_phase_for_collection(p):
                        """Normalize a phase value for collection.
                        
                        Maps:
                        - Integer 1 or string "1" -> 'mid'
                        - String 'mid' -> 'mid'
                        - Integer 0 or string "0" -> 'early'
                        - Integer 2 or string "2" -> 'late'
                        - Other values pass through as normalized strings
                        """
                        # Check if it's a string representation of an integer (e.g., "1", "0", "2")
                        if isinstance(p, str) and p.strip().isdigit():
                            # Convert string "1" to integer 1, then map to string
                            p_int = int(p.strip())
                            return phase_map_int.get(p_int, str(p_int).lower())
                        elif isinstance(p, str) and p.strip().lower() == 'int':
                            # String 'int' maps to 'mid' (since integer 1 maps to 'mid')
                            return 'mid'
                        elif isinstance(p, (int, np.integer)):
                            return phase_map_int.get(int(p), str(p).strip().lower())
                        else:
                            return str(p).strip().lower()
                    
                    if isinstance(phase_data, str):
                        # Single string phase - normalize and add it
                        available_phases.add(normalize_phase_for_collection(phase_data))
                    elif isinstance(phase_data, (int, np.integer)):
                        # Single integer phase - map to string
                        available_phases.add(normalize_phase_for_collection(phase_data))
                    elif hasattr(phase_data, '__iter__') and not isinstance(phase_data, str):
                        try:
                            # Array of phases - normalize each
                            normalized = [normalize_phase_for_collection(p) for p in np.unique(phase_data)]
                            available_phases.update(normalized)
                        except Exception:
                            pass
                    else:
                        # Scalar or other type - convert to string and normalize
                        try:
                            available_phases.add(normalize_phase_for_collection(phase_data))
                        except Exception:
                            pass
                
                available_stim_str = str(sorted(available_stim_values)) if available_stim_values else 'unknown'
                available_response_str = str(sorted(available_response_values)) if available_response_values else 'unknown'
                available_phases_str = str(sorted(available_phases)) if available_phases else 'unknown'
                
                error_msg = (
                    f"No valid trials found for task '{self.task_name}' in the selected datasets.\n"
                    f"  Requested: stim_values={self.stim_values}, response_values={self.response_values}, phases={self.phases}\n"
                    f"  Available in dataset: stim_values={available_stim_str}, "
                    f"response_values={available_response_str}, phases={available_phases_str}\n"
                    f"  Total candidate trials checked: {total_candidate_trials}\n"
                    f"  Trials filtered due to NaN: {nan_filtered_trials}\n"
                    f"  Please check that the requested values exist in the dataset."
                )
                if total_candidate_trials == 0:
                    error_msg += "\n  No trials matched stim/response criteria. Check that stim_data and response_data are correctly formatted arrays."
                elif nan_filtered_trials == total_candidate_trials and total_candidate_trials > 0:
                    error_msg += "\n  All trials were filtered due to NaN values. Check data quality."
            except Exception as e:
                # Fallback to simpler error message if collection fails
                error_msg = (
                    f"No valid trials found for task '{self.task_name}' in the selected datasets.\n"
                    f"  Requested: stim_values={self.stim_values}, response_values={self.response_values}, phases={self.phases}\n"
                    f"  Error while checking available values: {e}"
                )
            
            raise ValueError(error_msg)
        
        # Convert to numpy arrays
        # Check for shape consistency before converting
        if combined_neural_data:
            # Check shapes of all trials
            shapes = [trial.shape for trial in combined_neural_data]
            unique_shapes = set(shapes)
            if len(unique_shapes) > 1:
                logger.warning(f"Inconsistent trial shapes detected: {unique_shapes}")
                logger.warning(f"First 10 shapes: {shapes[:10]}")
                # Find the most common shape
                from collections import Counter
                shape_counts = Counter(shapes)
                most_common_shape = shape_counts.most_common(1)[0][0]
                logger.info(f"Most common shape: {most_common_shape}, filtering trials to match")
                
                # Filter to only include trials with the most common shape
                filtered_data = []
                filtered_labels = []
                filtered_indices = []
                filtered_mouse_ids = []
                for idx, trial in enumerate(combined_neural_data):
                    if trial.shape == most_common_shape:
                        filtered_data.append(trial)
                        filtered_labels.append(combined_labels[idx])
                        filtered_indices.append(combined_trial_indices[idx])
                        filtered_mouse_ids.append(combined_mouse_ids[idx])
                
                logger.info(f"Filtered from {len(combined_neural_data)} to {len(filtered_data)} trials with consistent shape")
                combined_neural_data = filtered_data
                combined_labels = filtered_labels
                combined_trial_indices = filtered_indices
                combined_mouse_ids = filtered_mouse_ids
        
        combined_neural_data = np.array(combined_neural_data)  # Shape: (n_trials, time_points, brain_areas)
        combined_labels = np.array(combined_labels)
        
        logger.info(f"Task '{self.task_name}': Combined {len(combined_labels)} valid trials from {len(dataset_indices)} datasets")
        
        # Create dynamic phase-to-label mapping for logging
        phase_label_map = {phase: idx for idx, phase in enumerate(self.phases)}
        logger.info(f"  Phase-to-label mapping: {phase_label_map}")
        
        # Log distribution for each phase
        for phase, label_idx in phase_label_map.items():
            count = np.sum(combined_labels == label_idx)
            logger.info(f"  {phase} trials (label={label_idx}): {count}")
        
        # Additional validation logging
        unique_labels, label_counts = np.unique(combined_labels, return_counts=True)
        label_distribution = dict(zip(unique_labels, label_counts))
        logger.info(f"  Label distribution: {label_distribution}")
        
        # Check for potential issues
        if len(unique_labels) == 1:
            single_label = unique_labels[0]
            single_phase = self.phases[single_label] if single_label < len(self.phases) else f"unknown({single_label})"
            logger.warning(f"⚠️  ALL TRIALS HAVE THE SAME LABEL: {single_label} (phase: {single_phase})")
            logger.warning(f"⚠️  This will cause training accuracy=1 and F1 issues!")
            logger.warning(f"⚠️  Check if phase data varies per trial or is constant per dataset")
        
        # Verify we have the expected number of classes
        expected_num_classes = len(self.phases)
        actual_num_classes = len(unique_labels)
        if actual_num_classes != expected_num_classes:
            logger.warning(f"⚠️  Expected {expected_num_classes} classes but found {actual_num_classes}")
            logger.warning(f"⚠️  Target phases: {self.phases}")
            logger.warning(f"⚠️  Actual labels: {sorted(unique_labels)}")
        
        if nan_filtered_trials > 0:
            logger.info(f"  Filtered out {nan_filtered_trials}/{total_candidate_trials} trials due to NaN values ({100.0 * nan_filtered_trials / total_candidate_trials:.2f}%)")
        
        logger.info(f"  Combined data shape: {combined_neural_data.shape}")
        logger.info(f"  Unique mice: {len(set(combined_mouse_ids))}")
        
        if combined_target_values is not None:
            return (combined_neural_data, combined_labels, np.array(combined_trial_indices),
                    combined_mouse_ids, np.array(combined_target_values, dtype=np.float32))
        return combined_neural_data, combined_labels, np.array(combined_trial_indices), combined_mouse_ids


# Column indices for data_table (assumes .mat structure: dff, zscore, stim, response, phase, mouse)
COLUMN_INDEX = {'dff': 0, 'zscore': 1, 'stim': 2, 'response': 3, 'phase': 4, 'mouse': 5}


def get_unique_column_values(
    dataset: WidefieldDataset,
    dataset_indices: List[int],
    column_name: str
) -> List[Any]:
    """Get unique values from a column across the given dataset indices."""
    if column_name not in COLUMN_INDEX:
        return []
    col_idx = COLUMN_INDEX[column_name]
    seen = set()
    for idx in dataset_indices:
        if idx >= dataset.data_table.shape[0]:
            continue
        data = dataset.data_table[idx, col_idx]
        if data is None:
            continue
        try:
            if hasattr(data, '__iter__') and not isinstance(data, str):
                flat = np.asarray(data).flatten()
                for v in flat:
                    v = v.item() if hasattr(v, 'item') else v
                    if isinstance(v, (int, float)) and np.isnan(v):
                        continue
                    seen.add(str(v).strip().lower() if isinstance(v, str) else v)
            else:
                v = data.item() if hasattr(data, 'item') else data
                if not (isinstance(v, (int, float)) and np.isnan(v)):
                    seen.add(str(v).strip().lower() if isinstance(v, str) else v)
        except (TypeError, ValueError):
            pass
    if column_name in ('stim', 'response'):
        out = []
        for x in seen:
            try:
                out.append(int(float(x)) if str(x).replace('.', '').isdigit() else x)
            except (ValueError, TypeError):
                out.append(x)
        return sorted(out)
    return sorted(list(seen))


def create_task_definition_from_flexible(
    target_column: str,
    target_values: Optional[List] = None,
    filters: Optional[Dict[str, List]] = None,
    task_mode: str = 'classification'
) -> TaskDefinition:
    """
    Create TaskDefinition from flexible, dataset-agnostic parameters.
    
    Maps to legacy TaskDefinition for backward compatibility.
    For genotype (target_column='mouse'), uses phases=['all'] and labels come from mouse IDs.
    For phase (target_column='phase'), uses target_values as phases.
    
    Args:
        target_column: Column to predict ('phase', 'mouse', 'stim', 'response')
        target_values: For classification, values to include (e.g. ['early','late'])
        filters: Dict of column -> values to include, e.g. {'stim': [1], 'response': [0,1]}
        task_mode: 'classification' or 'regression'
    
    Returns:
        TaskDefinition configured for the given params
    """
    filters = filters or {}
    stim_values = filters.get('stim', [1])
    response_values = filters.get('response', [0, 1])
    if target_column == 'mouse':
        phases = ['all']
    elif target_column == 'phase':
        if target_values:
            phases = [str(v).strip().lower() for v in target_values]
        else:
            phases = filters.get('phase', ['early', 'mid', 'late'])
    else:
        phases = filters.get('phase', ['early', 'late'])
    return TaskDefinition(
        stim_values=stim_values,
        response_values=response_values,
        phases=phases,
        task_name=f"{target_column}_{task_mode}"
    )


class WidefieldTrialDataset(Dataset):
    """
    PyTorch Dataset for individual trials.
    """
    
    def __init__(
        self,
        neural_data: np.ndarray,
        labels: np.ndarray,
        normalize_stats: Optional[Dict[str, np.ndarray]] = None,
        mouse_ids: Optional[np.ndarray] = None,
        region_pool: int = 1,
        time_pool: int = 1,
    ):
        """
        Initialize trial dataset with pre-filtered and combined data.
        
        Args:
            neural_data: Combined neural data array (n_trials, time_points, brain_areas)
            labels: Labels array (n_trials,)
            normalize_stats: Dict with 'mean' and 'std' for normalization
            mouse_ids: Optional array of mouse IDs (n_trials,)
            region_pool: Pool factor for brain areas (1=none, 2=avg pairs, etc.)
            time_pool: Pool factor for timepoints (1=none, 2=avg pairs, etc.)
        """
        self.neural_data = neural_data
        self.labels = labels
        self.normalize_stats = normalize_stats
        self.mouse_ids = mouse_ids if mouse_ids is not None else np.array([None] * len(labels))
        self.region_pool = max(1, int(region_pool))
        self.time_pool = max(1, int(time_pool))
        
        if len(neural_data) != len(labels):
            raise ValueError(f"Neural data and labels must have same length: {len(neural_data)} vs {len(labels)}")
        if len(self.mouse_ids) != len(labels):
            raise ValueError(f"Mouse IDs must have same length as labels: {len(self.mouse_ids)} vs {len(labels)}")
        
        logger.info(f"Created dataset with {len(self.labels)} trials")
        logger.info(f"  Data shape: {self.neural_data.shape}")
        if len(self.neural_data) > 0:
            logger.info(f"  First trial shape: {self.neural_data[0].shape}")
            logger.info(f"  Expected format: (trials, timepoints, brain_areas)")
            logger.info(f"  If shape is (trials, brain_areas, timepoints), it will be transposed in __getitem__")
        logger.info(f"  Label distribution: {dict(zip(*np.unique(self.labels, return_counts=True)))}")
        if mouse_ids is not None:
            logger.info(f"  Unique mice: {len(set(self.mouse_ids))}")
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get a single trial.
        
        Args:
            idx: Index of the trial
            
        Returns:
            Tuple of (trial_data, label)
        """
        # Extract single trial
        # neural_data should be (trials, timepoints, brain_areas)
        # But check if it's actually (trials, brain_areas, timepoints)
        trial_data = self.neural_data[idx].copy()
        label = self.labels[idx]
        mouse_id = str(self.mouse_ids[idx]) if self.mouse_ids[idx] is not None else "unknown"
        
        # Log shape for debugging
        if idx == 0:
            logger.info(f"Trial 0 raw shape: {trial_data.shape}, full neural_data shape: {self.neural_data.shape}")
        
        # Check if data is in wrong format
        # Expected: (timepoints, brain_areas) from (trials, timepoints, brain_areas)
        # If we get (brain_areas, timepoints), the data was stored as (trials, brain_areas, timepoints)
        # Typical: timepoints ~30-50, brain_areas ~50-2000 (varies by dataset)
        # Rule: If first dim is much larger than second dim AND first dim > 100, likely (brain_areas, timepoints)
        #       If first dim < 100 and second dim > 50, likely (timepoints, brain_areas) - CORRECT
        
        n_timepoints_dim = trial_data.shape[0]
        n_brain_areas_dim = trial_data.shape[1]
        
        # Check if shape suggests (brain_areas, timepoints) format
        # This happens when first dim is large (>100) and second dim is small (<100)
        # AND first dim is much larger than second dim
        if n_timepoints_dim > 100 and n_brain_areas_dim < 100 and n_timepoints_dim > n_brain_areas_dim * 2:
            # Likely (brain_areas, timepoints) - transpose to (timepoints, brain_areas)
            if idx < 5:  # Log first few
                logger.warning(f"Trial {idx}: Detected wrong format {trial_data.shape} (likely brain_areas, timepoints)")
                logger.warning(f"  Transposing to (timepoints, brain_areas)")
            trial_data = trial_data.T
            n_timepoints_dim, n_brain_areas_dim = trial_data.shape[0], trial_data.shape[1]
        elif n_timepoints_dim < 100 and n_brain_areas_dim > 50:
            # Likely (timepoints, brain_areas) - correct format, no transpose needed
            if idx == 0:
                logger.info(f"Trial {idx}: Correct format {trial_data.shape} (timepoints={n_timepoints_dim}, brain_areas={n_brain_areas_dim})")
        else:
            # Ambiguous - assume correct format (timepoints, brain_areas)
            if idx == 0:
                logger.info(f"Trial {idx}: Shape {trial_data.shape} - assuming (timepoints={n_timepoints_dim}, brain_areas={n_brain_areas_dim})")
        
        # Handle NaN values: convert to 0 (following reference preprocessing)
        # Reference: x = torch.from_numpy(np.nan_to_num(self.data_calcium[idx, ...]))
        # Note: Trials with NaN should be filtered out during data loading,
        # but this provides an additional safety measure consistent with reference code
        trial_data = np.nan_to_num(trial_data, nan=0.0)
        
        # Apply region and time pooling (configurable)
        # trial_data: (timepoints, brain_areas)
        n_timepoints = trial_data.shape[0]
        n_areas = trial_data.shape[1]
        
        if self.region_pool > 1 and n_areas >= self.region_pool:
            n_pooled = n_areas // self.region_pool
            pooled_areas = []
            for i in range(n_pooled):
                start = i * self.region_pool
                end = start + self.region_pool
                pooled_areas.append(trial_data[:, start:end].mean(axis=1))
            trial_data = np.stack(pooled_areas, axis=1)  # (timepoints, n_pooled)
            n_areas = trial_data.shape[1]
            if idx == 0:
                logger.info(f"Trial {idx}: Region pool {self.region_pool}: {n_areas * self.region_pool} -> {n_areas} areas")
        
        if self.time_pool > 1 and n_timepoints >= self.time_pool:
            n_pooled_t = n_timepoints // self.time_pool
            pooled_time = []
            for i in range(n_pooled_t):
                start = i * self.time_pool
                end = start + self.time_pool
                pooled_time.append(trial_data[start:end, :].mean(axis=0))
            trial_data = np.stack(pooled_time, axis=0)  # (n_pooled_t, n_areas)
            n_timepoints = trial_data.shape[0]
            if idx == 0:
                logger.info(f"Trial {idx}: Time pool {self.time_pool}: {n_timepoints * self.time_pool} -> {n_timepoints} timepoints")
        
        if idx == 0:
            logger.info(f"Final shape: trial_data.shape={trial_data.shape}, timepoints={n_timepoints}, regions={n_areas}")
        
        # Apply scaling/normalization
        if self.normalize_stats is not None:
            if self.normalize_stats['normalization_type'] == 'scale_20':
                # Simple *20 scaling (following reference preprocessing)
                # Reference: parcel_data[transitions[i]-2:transitions[i]+12,:]*20
                trial_data = trial_data * 20.0
            elif self.normalize_stats['normalization_type'] == 'max':
                # Legacy max normalization (for backward compatibility)
                trial_data = trial_data / self.normalize_stats['max_values'][None, :]
            elif self.normalize_stats['normalization_type'] == 'robust_scaler':
                # RobustScaler: (x - median) / IQR
                median_vals = self.normalize_stats['median_values'][None, :]  # (1, brain_areas)
                iqr_vals = self.normalize_stats['iqr_values'][None, :]  # (1, brain_areas)
                trial_data = (trial_data - median_vals) / iqr_vals
            elif self.normalize_stats['normalization_type'] == 'percentile_threshold':
                # For z-score: threshold at 95th and 5th percentiles (clip extreme values)
                percentile_95_vals = self.normalize_stats['percentile_95_values'][None, :]
                percentile_5_vals = self.normalize_stats['percentile_5_values'][None, :]
                trial_data = np.clip(trial_data, percentile_5_vals, percentile_95_vals)
            elif self.normalize_stats['normalization_type'] == 'none':
                # No normalization applied (legacy support)
                pass
            else:
                raise ValueError(f"Unknown normalization type: {self.normalize_stats['normalization_type']}")
        
        # Return (timepoints, brain_areas) - model expects (batch, time_points, n_brain_areas)
        # Support both classification (int labels) and regression (float labels)
        if np.issubdtype(type(label), np.floating) or isinstance(label, float):
            label_tensor = torch.tensor(label, dtype=torch.float32)
        else:
            label_tensor = torch.LongTensor([int(label)])[0]
        return torch.FloatTensor(trial_data), label_tensor, mouse_id


def compute_normalization_stats(
    neural_data: np.ndarray,
    data_type: str = 'dff'
) -> Dict[str, np.ndarray]:
    """
    Compute normalization statistics per brain area across all trials and animals.
    
    This function computes normalization statistics (median, IQR for dff, or percentiles for zscore)
    for each brain area independently. The statistics are computed across all trials and timepoints,
    ensuring consistent normalization across animals.
    
    Args:
        neural_data: Neural data array (n_trials, time_points, brain_areas)
                     Should include data from all animals (both train and validation) for
                     consistent normalization across animals.
        data_type: Either 'dff' or 'zscore' - determines normalization strategy
        
    Returns:
        Dictionary with normalization parameters based on data_type.
        For 'dff': returns median_values and iqr_values per brain area.
        For 'zscore': returns percentile_95_values and percentile_5_values per brain area.
    """
    # Apply brain area averaging first (if needed)
    if neural_data.shape[2] == 82:
        # Average every pair of adjacent brain areas
        averaged_data = []
        for i in range(0, 82, 2):
            avg_area = (neural_data[:, :, i] + neural_data[:, :, i + 1]) / 2
            averaged_data.append(avg_area)
        neural_data = np.stack(averaged_data, axis=2)  # (n_trials, time_points, 41)
    
    if data_type == 'dff':
        # For dff: use RobustScaler normalization (median and IQR)
        # RobustScaler: (x - median) / IQR, where IQR = Q3 - Q1
        median_stats = np.median(neural_data, axis=(0, 1))  # (brain_areas,) - median across trials and time
        q1_stats = np.percentile(neural_data, 25, axis=(0, 1))  # (brain_areas,) - 25th percentile
        q3_stats = np.percentile(neural_data, 75, axis=(0, 1))  # (brain_areas,) - 75th percentile
        iqr_stats = q3_stats - q1_stats  # (brain_areas,) - interquartile range
        
        # Avoid division by zero: if IQR is 0, use a small epsilon or fallback to std
        iqr_stats = np.where(iqr_stats == 0, np.std(neural_data, axis=(0, 1)), iqr_stats)
        iqr_stats = np.where(iqr_stats == 0, 1.0, iqr_stats)  # Final fallback to 1.0
        
        logger.info(f"Computed RobustScaler normalization stats for {len(median_stats)} brain areas (dff)")
        logger.info(f"  Median range: [{median_stats.min():.6f}, {median_stats.max():.6f}]")
        logger.info(f"  IQR range: [{iqr_stats.min():.6f}, {iqr_stats.max():.6f}]")
        
        return {
            'normalization_type': 'robust_scaler',
            'median_values': median_stats,
            'iqr_values': iqr_stats
        }
    
    elif data_type == 'zscore':
        # For z-score: use 95th and 5th percentile thresholding instead of normalization
        # Compute 95th and 5th percentiles across trials and time for each brain area
        percentile_95_stats = np.percentile(neural_data, 95, axis=(0, 1))  # (brain_areas,)
        percentile_5_stats = np.percentile(neural_data, 5, axis=(0, 1))   # (brain_areas,)
        
        # Also compute some statistics for logging
        max_vals = np.max(neural_data, axis=(0, 1))
        min_vals = np.min(neural_data, axis=(0, 1))
        
        logger.info(f"Computed percentile thresholding stats for {len(percentile_95_stats)} brain areas (zscore)")
        logger.info(f"  95th percentile range: [{percentile_95_stats.min():.6f}, {percentile_95_stats.max():.6f}]")
        logger.info(f"  5th percentile range: [{percentile_5_stats.min():.6f}, {percentile_5_stats.max():.6f}]")
        logger.info(f"  Original data range: [{min_vals.min():.6f}, {max_vals.max():.6f}]")
        
        return {
            'normalization_type': 'percentile_threshold',
            'percentile_95_values': percentile_95_stats,
            'percentile_5_values': percentile_5_stats
        }
    
    else:
        raise ValueError(f"Unknown data_type: {data_type}. Must be 'dff' or 'zscore'")


def create_data_loaders(
    dataset: WidefieldDataset,
    task_definition: TaskDefinition,
    train_indices: List[int],
    val_indices: List[int],
    data_type: str = 'dff',
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, Dict[str, np.ndarray]]:
    """
    Create training and validation data loaders.
    
    Args:
        dataset: WidefieldDataset instance
        task_definition: TaskDefinition instance
        train_indices: Training dataset indices (row indices)
        val_indices: Validation dataset indices (row indices)
        data_type: Either 'dff' or 'zscore'
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, normalization_stats)
    """
    # Filter and combine training datasets
    train_neural_data, train_labels, train_trial_indices, train_mouse_ids = task_definition.filter_trials(
        dataset, train_indices, data_type
    )
    
    # Filter and combine validation datasets
    val_neural_data, val_labels, val_trial_indices, val_mouse_ids = task_definition.filter_trials(
        dataset, val_indices, data_type
    )
    
    logger.info(f"Combined data:")
    logger.info(f"  Train: {train_neural_data.shape} from {len(set(train_mouse_ids))} mice")
    logger.info(f"  Val: {val_neural_data.shape} from {len(set(val_mouse_ids))} mice")
    
    # Use simple *20 scaling instead of normalization (following reference preprocessing)
    # Reference: parcel_data[transitions[i]-2:transitions[i]+12,:]*20
    # Create dummy normalization stats dict for compatibility (not used, scaling is applied in dataset)
    norm_stats = {'normalization_type': 'scale_20'}
    
    # Create datasets with mouse IDs
    train_dataset = WidefieldTrialDataset(train_neural_data, train_labels, norm_stats, train_mouse_ids)
    val_dataset = WidefieldTrialDataset(val_neural_data, val_labels, norm_stats, val_mouse_ids)
    
    # Create balanced batch sampler for training
    try:
        balanced_sampler = BalancedBatchSampler(train_dataset, batch_size)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=balanced_sampler,
            num_workers=num_workers,
            pin_memory=True
        )
        logger.info(f"Using balanced batch sampler for training with {len(balanced_sampler)} batches")
    except ValueError as e:
        logger.warning(f"Could not create balanced sampler: {e}")
        logger.warning("Falling back to regular shuffled sampling")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created data loaders - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    return train_loader, val_loader, norm_stats

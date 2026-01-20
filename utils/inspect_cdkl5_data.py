#!/usr/bin/env python3
"""
Inspect CDKL5 MATLAB data file structure.
"""

import h5py
import numpy as np
from pathlib import Path

def inspect_cdkl5_file(mat_file_path):
    """Inspect CDKL5 MATLAB file structure."""
    mat_file = Path(mat_file_path)
    
    if not mat_file.exists():
        print(f"File not found: {mat_file}")
        return
    
    print(f"Loading MATLAB file: {mat_file}")
    print()
    
    with h5py.File(str(mat_file), 'r') as f:
        print("Variables in file:", list(f.keys()))
        print()
        
        def inspect_struct_array(struct_name):
            """Inspect a MATLAB struct array."""
            if struct_name not in f:
                return
            
            print("=" * 60)
            print(f"{struct_name}")
            print("=" * 60)
            struct_ref = f[struct_name]
            
            if isinstance(struct_ref, h5py.Dataset):
                # It's a reference array - MATLAB struct arrays are stored as reference arrays
                refs = struct_ref[0]  # Get the array of references
                n_animals = len(refs)
                print(f"Number of animals: {n_animals}")
                
                if n_animals > 0:
                    # Read first animal structure
                    try:
                        # Access via #refs# group using the reference ID
                        refs_group = f['#refs#']
                        
                        # The reference is stored as an ID, need to access via #refs#
                        first_ref_id = refs[0]
                        first_animal_ref = refs_group[str(first_ref_id)]
                        
                        if isinstance(first_animal_ref, h5py.Group):
                            field_names = list(first_animal_ref.keys())
                            print(f"Fields in first animal: {len(field_names)}")
                            print(f"Field names: {field_names}")
                            
                            # Inspect key fields
                            for field_name in field_names[:15]:
                                try:
                                    field_ref = first_animal_ref[field_name]
                                    if isinstance(field_ref, h5py.Dataset):
                                        print(f"  {field_name}: shape={field_ref.shape}, dtype={field_ref.dtype}")
                                        # Show a sample if it's small
                                        if field_ref.size < 100:
                                            try:
                                                sample = field_ref[:]
                                                print(f"    Sample: {sample}")
                                            except:
                                                pass
                                    elif isinstance(field_ref, h5py.Group):
                                        sub_keys = list(field_ref.keys())
                                        print(f"  {field_name}: Group with {len(sub_keys)} keys")
                                        if len(sub_keys) <= 10:
                                            print(f"    Keys: {sub_keys}")
                                    elif hasattr(field_ref, 'ref'):
                                        # It's a reference
                                        ref_val = field_ref.ref
                                        if ref_val:
                                            try:
                                                referenced = refs_group[str(ref_val)]
                                                print(f"  {field_name}: Reference -> {type(referenced)}")
                                            except:
                                                print(f"  {field_name}: Reference (could not follow)")
                                except Exception as e:
                                    print(f"  {field_name}: Error reading - {e}")
                    except Exception as e:
                        print(f"Error reading first animal: {e}")
                        import traceback
                        traceback.print_exc()
            print()
        
        # Inspect both structs
        inspect_struct_array('cdkl5_m_wt_struct')
        inspect_struct_array('cdkl5_m_mut_struct')

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        mat_file_path = sys.argv[1]
    else:
        mat_file_path = "/Users/josueortegacaro/Documents/rachel_cdkl5/cdkl5_data_for_josue_w_states.mat"
    
    inspect_cdkl5_file(mat_file_path)

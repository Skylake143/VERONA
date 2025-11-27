#!/usr/bin/env python3
"""
Script to extract ONNX files from TL* folders and rename them to store in TransferLearned folder.
"""

import os
import shutil
from pathlib import Path

def extract_onnx_files():
    """
    Extract ONNX files from folders beginning with 'TL' and rename them
    to the first three letters of the directory name (TL1, TL2, etc.).
    Store them in the TransferLearned folder.
    """
    # Get the current directory (should be CNNYangBig)
    current_dir = Path.cwd()
    
    # Create TransferLearned directory if it doesn't exist
    transfer_learned_dir = current_dir / "TransferLearned"
    transfer_learned_dir.mkdir(exist_ok=True)
    
    print(f"Looking for TL* folders in: {current_dir}")
    print(f"Target directory: {transfer_learned_dir}")
    
    # Find all directories starting with "TL"
    tl_folders = [d for d in current_dir.iterdir() if d.is_dir() and d.name.startswith("TL")]
    
    if not tl_folders:
        print("No folders starting with 'TL' found!")
        return
    
    print(f"Found {len(tl_folders)} TL folders: {[f.name for f in tl_folders]}")
    
    extracted_count = 0
    
    for tl_folder in sorted(tl_folders):
        # Get the first three letters of the folder name
        new_name_prefix = tl_folder.name[:3]
        
        print(f"\nProcessing folder: {tl_folder.name}")
        
        # Find all ONNX files in the folder (recursively)
        onnx_files = list(tl_folder.rglob("*.onnx"))
        
        if not onnx_files:
            print(f"  No ONNX files found in {tl_folder.name}")
            continue
        
        print(f"  Found {len(onnx_files)} ONNX file(s)")
        
        for i, onnx_file in enumerate(onnx_files):
            # Create new filename
            if len(onnx_files) == 1:
                new_filename = f"{new_name_prefix}.onnx"
            else:
                new_filename = f"{new_name_prefix}_{i+1}.onnx"
            
            # Full path for the new file
            new_file_path = transfer_learned_dir / new_filename
            
            # Copy the file
            try:
                shutil.copy2(onnx_file, new_file_path)
                print(f"  Copied: {onnx_file.name} -> {new_filename}")
                extracted_count += 1
            except Exception as e:
                print(f"  Error copying {onnx_file.name}: {e}")
    
    print(f"\n✅ Extraction complete! {extracted_count} ONNX files copied to TransferLearned folder.")

if __name__ == "__main__":
    extract_onnx_files()
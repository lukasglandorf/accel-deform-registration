#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download OCTA example data from Google Drive.

This script downloads the example OCTA volumes for testing registration.

Usage:
    python scripts/download_octa_data.py

Requirements:
    pip install gdown
"""

import os
import sys
from pathlib import Path

# File IDs from the Google Drive folder
# https://drive.google.com/drive/folders/1MrrfGk3sAkXkOEn5Ys2ZpH67fhJkna5v
FOLDER_ID = "1MrrfGk3sAkXkOEn5Ys2ZpH67fhJkna5v"

# Output directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "octa"


def main():
    """Download OCTA example data."""
    try:
        import gdown
    except ImportError:
        print("Error: gdown is required. Install with: pip install gdown")
        sys.exit(1)
    
    # Create output directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading OCTA example data to: {DATA_DIR}")
    print(f"Google Drive folder: https://drive.google.com/drive/folders/{FOLDER_ID}")
    print()
    
    # Download the entire folder
    url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
    
    try:
        gdown.download_folder(url, output=str(DATA_DIR), quiet=False)
        print()
        print("Download complete!")
        print()
        
        # Check what was downloaded
        files = list(DATA_DIR.glob("*.nii.gz"))
        if files:
            print("Downloaded files:")
            for f in files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  - {f.name} ({size_mb:.1f} MB)")
        else:
            print("Warning: No .nii.gz files found after download.")
            print("Please check the Google Drive folder manually.")
            
    except Exception as e:
        print(f"Error downloading: {e}")
        print()
        print("Please download manually from:")
        print(f"  https://drive.google.com/drive/folders/{FOLDER_ID}")
        print()
        print(f"And place the files in: {DATA_DIR}")
        sys.exit(1)
    
    # Verify expected files
    source_file = DATA_DIR / "sourceExample.nii.gz"
    target_file = DATA_DIR / "targetExample.nii.gz"
    
    print()
    if source_file.exists() and target_file.exists():
        print("✓ All required files present!")
        print()
        print("You can now run the OCTA tests:")
        print("  python tests/run_tests.py --octa")
    else:
        print("Warning: Expected files not found:")
        if not source_file.exists():
            print(f"  Missing: {source_file.name}")
        if not target_file.exists():
            print(f"  Missing: {target_file.name}")


if __name__ == "__main__":
    main()

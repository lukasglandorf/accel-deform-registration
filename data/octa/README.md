# OCTA Example Data

This directory contains example OCTA (Optical Coherence Tomography Angiography) volumes for testing the registration algorithms.

## Required Files

Place the following files in this directory:

- `sourceExample.nii.gz` - The source/moving volume to be registered
- `targetExample.nii.gz` - The target/fixed volume (reference)

## Download

Download the example data from Google Drive:
https://drive.google.com/drive/folders/1MrrfGk3sAkXkOEn5Ys2ZpH67fhJkna5v

Or use the download script:

```bash
cd /path/to/accel-deform-registration
python scripts/download_octa_data.py
```

## File Format

- Format: NIfTI (.nii.gz)
- Dimensions: 3D volumes (Z, Y, X)
- The volumes should be pre-aligned approximately (rigid alignment)

## Usage

Once the data is in place, run tests with:

```bash
python tests/run_tests.py --octa
```

# OCTA Test Data Directory

Place your OCTA test images here for registration testing.

## Expected File Names

For **2D MIP images** (any of these formats):
- `target_2d.tif` or `target_2d.png` - Target/fixed 2D image
- `moving_2d.tif` or `moving_2d.png` - Moving/source 2D image

For **3D volumes** (any of these formats):
- `target_3d.nii.gz` or `target_3d.tif` - Target/fixed 3D volume
- `moving_3d.nii.gz` or `moving_3d.tif` - Moving/source 3D volume

## Supported Formats

- **2D**: TIFF (.tif, .tiff), PNG (.png), JPEG (.jpg)
- **3D**: NIfTI (.nii, .nii.gz), 3D TIFF (.tif, .tiff)

## Notes

- Images will be automatically normalized to [0, 1] range
- Grayscale images work best
- 3D volumes should have shape (Z, Y, X)
- If only 2D or only 3D data is provided, only those tests will run

## Example Usage

1. Copy your OCTA MIP images:
   ```
   cp /path/to/your/mip_target.tif tests/data/octa/target_2d.tif
   cp /path/to/your/mip_moving.tif tests/data/octa/moving_2d.tif
   ```

2. Run the tests with OCTA data:
   ```
   python tests/run_tests.py --octa
   ```

# Data directory

Place test data files here:
- `fixed_mip.npy` - Fixed (target) MIP image for 2D demo
- `moving_mip.npy` - Moving (source) MIP image for 2D demo
- `fixed_volume.npy` - Fixed (target) 3D volume for 3D demo
- `moving_volume.npy` - Moving (source) 3D volume for 3D demo

If data is not present, the demo scripts will generate synthetic phantoms.

## Downloading Sample Data

Sample OCTA data can be downloaded from Google Drive (TODO: add link).

Or run:
```bash
python examples/demo_2d_mip.py --download
python examples/demo_3d_pyramid.py --download
```

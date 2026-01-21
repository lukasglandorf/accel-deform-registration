# Accelerated Deformable Registration

GPU-accelerated Free-Form Deformation (FFD) registration for 2D and 3D images using PyTorch.

## Features

- **B-spline FFD registration** for both 2D images and 3D volumes
- **GPU acceleration** via PyTorch (CUDA, MPS, or CPU fallback)
- **Per-axis grid spacing** for anisotropic data (e.g., different resolution per axis)
- **Multiple similarity metrics**: Correlation, MAE, Mutual Information, Local NCC
- **Regularization**: Configurable smoothness and bending energy penalties
- **Pyramid registration** for coarse-to-fine alignment via image downsampling
- **Multiscale registration** for coarse-to-fine control point refinement at full resolution
- **Loss comparison utility**: Compare all loss functions with single/multiscale approaches
- **Boundary layer** option to reduce edge artifacts

## Installation

### From source (recommended for development)

```bash
git clone https://github.com/lukasglandorf/accel-deform-registration.git
cd accel-deform-registration
pip install -e ".[all]"
```

### Basic installation

```bash
pip install -e .
```

### With optional dependencies

```bash
# For running examples with visualization
pip install -e ".[examples]"

# For development and testing
pip install -e ".[dev]"

# For optimized MONAI loss functions (recommended for large images)
pip install -e ".[monai]"

# Install everything
pip install -e ".[all]"
```

## Quick Start

### 2D Registration

```python
import numpy as np
from accel_deform_registration import register_ffd_2d, apply_ffd_2d

# Load your images
moving = np.load("moving_image.npy")
fixed = np.load("fixed_image.npy")

# Register
displacement, ctrl_disps, ctrl_pos, mask, info = register_ffd_2d(
    moving, fixed,
    grid_spacing=100,
    n_iterations=2000,
)

print(f"Correlation: {info['initial_correlation']:.4f} → {info['final_correlation']:.4f}")

# Apply to another image
warped = apply_ffd_2d(moving, displacement)
```

### 3D Registration

```python
from accel_deform_registration import register_ffd_3d, apply_ffd_3d

# Load your volumes (Z, Y, X)
moving_vol = np.load("moving_volume.npy")
fixed_vol = np.load("fixed_volume.npy")

# Register with uniform spacing
displacement, ctrl_disps, ctrl_pos, mask, info = register_ffd_3d(
    moving_vol, fixed_vol,
    grid_spacing=40,
    n_iterations=400,
)

# Apply
warped_vol = apply_ffd_3d(moving_vol, displacement)
```

### Per-Axis Grid Spacing

For anisotropic data (different resolution per axis), use per-axis spacing:

```python
# 2D: (spacing_y, spacing_x)
displacement, _, _, _, info = register_ffd_2d(
    moving, fixed,
    grid_spacing=(100, 80),  # Y=100, X=80 pixels
)

# 3D: (spacing_z, spacing_y, spacing_x) - useful for anisotropic volumes
displacement, _, _, _, info = register_ffd_3d(
    moving_vol, fixed_vol,
    grid_spacing=(20, 40, 40),  # Finer spacing in Z for thick slices
)

# Control points are centered on the image with exact spacing
print(f"Grid offset: {info['grid_offset']}")  # Starting offset from origin
print(f"Grid spacing: {info['grid_spacing']}")  # Actual spacing used
```

### Using Different Loss Functions

```python
from accel_deform_registration import register_ffd_2d
from accel_deform_registration.losses import MAELoss, MutualInformationLoss, get_loss_function

# Using MAE loss
displacement, *_ = register_ffd_2d(
    moving, fixed,
    loss_fn=MAELoss(),
)

# Using Mutual Information (good for multi-modal registration)
displacement, *_ = register_ffd_2d(
    moving, fixed,
    loss_fn=MutualInformationLoss(num_bins=64),
)

# Using MONAI's optimized MI (faster, requires: pip install monai)
displacement, *_ = register_ffd_2d(
    moving, fixed,
    loss_fn=get_loss_function('monai_mi', num_bins=64),
)
```

## Regularization Weights

The registration functions accept two key regularization parameters that control the trade-off between data fidelity and deformation smoothness:

### `smooth_weight` - Displacement Magnitude Penalty

Controls how much to penalize large displacements (L2 norm of displacement vectors).

| Value | Effect |
|-------|--------|
| 0.001 | Very flexible, allows large deformations |
| 0.005 | **Default for 2D** - balanced |
| 0.01 | **Default for 3D** - balanced |
| 0.05-0.1 | More rigid, prevents unrealistic deformations |

### `bending_weight` - Bending Energy Penalty

Controls smoothness by penalizing high-frequency variations (second derivatives) in the deformation field.

| Value | Effect |
|-------|--------|
| 0.001-0.005 | Allows sharper local deformations |
| 0.01 | **Default** - smooth, physically plausible |
| 0.05-0.1 | Very smooth deformations |

### Examples

```python
from accel_deform_registration import register_ffd_2d

# Default registration (balanced)
displacement, *_ = register_ffd_2d(moving, fixed)

# More rigid registration (less deformation allowed)
displacement, *_ = register_ffd_2d(
    moving, fixed,
    smooth_weight=0.1,
    bending_weight=0.1,
)

# More flexible registration (larger deformations allowed)
displacement, *_ = register_ffd_2d(
    moving, fixed,
    smooth_weight=0.001,
    bending_weight=0.005,
)
```

## Pyramid Registration (Coarse-to-Fine)

Image pyramid registration downsamples images and uses progressively finer grids:

```python
from accel_deform_registration import pyramid_register_2d, pyramid_register_3d

# 2D pyramid registration
displacement, info = pyramid_register_2d(
    moving, fixed,
    levels=[4, 2, 1],  # Downsample factors
    grid_spacings=[50, 25, 12],
    iterations_per_level=[500, 300, 200],
)

# 3D pyramid registration
displacement, info = pyramid_register_3d(
    moving_vol, fixed_vol,
    levels=[4, 2, 1],
    grid_spacings=[40, 20, 10],
    iterations_per_level=[200, 150, 100],
)
```

## Multiscale Registration (Control Point Refinement)

Multiscale registration keeps images at full resolution but refines control points from coarse to fine. This produces a list of displacement fields that can be composed:

```python
from accel_deform_registration import multiscale_register_2d, apply_transforms_2d
from accel_deform_registration import multiscale_register_3d, apply_transforms_3d

# 2D multiscale - coarse to fine control grid
displacements, info = multiscale_register_2d(
    moving, fixed,
    grid_spacings=[200, 100, 50],  # Coarse → fine
    iterations_per_scale=[1000, 500, 300],
)

# Apply all displacements in sequence
warped = apply_transforms_2d(moving, displacements)

# 3D multiscale
displacements, info = multiscale_register_3d(
    moving_vol, fixed_vol,
    grid_spacings=[80, 40, 20],
    iterations_per_scale=[300, 200, 100],
)
warped_vol = apply_transforms_3d(moving_vol, displacements)
```

## Comparing Loss Functions

Use `compare_loss_functions` to test all available loss functions with both single and multiscale approaches:

```python
from accel_deform_registration import compare_loss_functions

# Compare all losses on your data (works for 2D or 3D)
results = compare_loss_functions(
    moving, fixed,
    grid_spacing=100,           # Single-scale grid spacing
    n_iterations=1000,          # Single-scale iterations
    multiscale_spacings=[200, 100, 50],  # Multiscale grid spacings
    multiscale_iterations=[500, 300, 200],
)

# Results are sorted by final correlation (best first)
for r in results:
    print(f"{r.name}: {r.initial_correlation:.4f} → {r.final_correlation:.4f}")
```

## Loss Functions

| Loss Function | Description | Best For |
|---------------|-------------|----------|
| `CorrelationLoss` | Pearson correlation (default) | Same-modality images |
| `MAELoss` | Mean Absolute Error (L1) | General purpose |
| `MutualInformationLoss` | MI with soft histogram | Multi-modal registration |
| `MonaiGlobalMILoss` | MONAI's optimized Global MI | Multi-modal (fast, GPU) |
| `MonaiLocalNCCLoss` | MONAI's Local NCC | Robust same-modality |

### Using the Loss Function Factory

```python
from accel_deform_registration.losses import get_loss_function

# Available names:
# - 'correlation', 'corr': CorrelationLoss
# - 'mae', 'l1': MAELoss  
# - 'mi', 'mutual_information': MutualInformationLoss
# - 'monai_mi', 'monai_global_mi': MonaiGlobalMILoss (requires MONAI)
# - 'lncc', 'local_ncc', 'monai_lncc': MonaiLocalNCCLoss (requires MONAI)

loss_fn = get_loss_function('mi', num_bins=64)
loss_fn = get_loss_function('monai_mi', num_bins=64)  # Faster, requires MONAI
```

## Test Data (OCTA)

Example OCTA volumes for testing are available on Google Drive:

**Download:** https://drive.google.com/drive/folders/1MrrfGk3sAkXkOEn5Ys2ZpH67fhJkna5v

Required files:
- `sourceExample.nii.gz` - Moving/source volume
- `targetExample.nii.gz` - Fixed/target volume

### Download Script

```bash
# Install gdown first
pip install gdown

# Run download script
python scripts/download_octa_data.py
```

Or manually download and place files in `data/octa/`.

### Run OCTA Tests

```bash
python tests/run_tests.py --octa
```

## Examples

See the `examples/` directory for complete examples:

- `demo_2d_mip.py` - 2D registration on MIP images
- `demo_3d_pyramid.py` - 3D pyramid registration on volumes
- `quick_test.py` - Quick validation test

## Running Tests

```bash
# Run comprehensive test suite
python tests/run_tests.py

# Run specific test groups
python tests/run_tests.py --losses     # Loss function tests only
python tests/run_tests.py --2d         # 2D registration tests only
python tests/run_tests.py --3d         # 3D registration tests only
python tests/run_tests.py --octa       # OCTA data tests (if data present)

# Run with pytest
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=accel_deform_registration
```

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `register_ffd_2d()` | 2D FFD registration |
| `register_ffd_3d()` | 3D FFD registration |
| `apply_ffd_2d()` | Apply 2D displacement field |
| `apply_ffd_3d()` | Apply 3D displacement field |

### Pyramid Registration (Image Downsampling)

| Function | Description |
|----------|-------------|
| `pyramid_register_2d()` | Multi-resolution 2D registration with downsampling |
| `pyramid_register_3d()` | Multi-resolution 3D registration with downsampling |

### Multiscale Registration (Control Point Refinement)

| Function | Description |
|----------|-------------|
| `multiscale_register_2d()` | Full-resolution 2D with coarse-to-fine control points |
| `multiscale_register_3d()` | Full-resolution 3D with coarse-to-fine control points |
| `apply_transforms_2d()` | Apply sequence of 2D displacement fields |
| `apply_transforms_3d()` | Apply sequence of 3D displacement fields |

### Loss Functions

| Class | Description |
|-------|-------------|
| `CorrelationLoss` | Negative Pearson correlation |
| `MAELoss` | Mean Absolute Error |
| `MutualInformationLoss` | Mutual Information (soft histogram) |
| `MonaiGlobalMILoss` | MONAI's optimized Global MI (requires MONAI) |
| `MonaiLocalNCCLoss` | MONAI's Local NCC (requires MONAI) |

### Utilities

| Function | Description |
|----------|-------------|
| `create_2d_grid_image()` | Visualize 2D control grid |
| `create_3d_grid_image()` | Visualize 3D control grid (MIP) |
| `shepp_logan_2d()` | Generate 2D Shepp-Logan phantom |
| `shepp_logan_3d()` | Generate 3D Shepp-Logan phantom |
| `get_loss_function()` | Create loss function by name |
| `compare_loss_functions()` | Compare all loss functions on your data |

## Citation

If you use this package in your research, please cite:

```bibtex
@software{accel_deform_registration,
  author = {Glandorf, Lukas},
  title = {Accelerated Deformable Registration},
  year = {2026},
  url = {https://github.com/lukasglandorf/accel-deform-registration}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

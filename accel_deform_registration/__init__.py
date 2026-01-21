# -*- coding: utf-8 -*-
"""
Accelerated Deformable Registration Package.

GPU-accelerated Free-Form Deformation (FFD) registration for 2D and 3D images.

Main Functions
--------------
- register_ffd_2d: 2D B-spline FFD registration
- register_ffd_3d: 3D B-spline FFD registration
- apply_ffd_2d: Apply 2D displacement field
- apply_ffd_3d: Apply 3D displacement field

Multi-Resolution Registration
-----------------------------
Image Pyramid (downsample images):
- pyramid_register_2d: Multi-resolution 2D registration with downsampling
- pyramid_register_3d: Multi-resolution 3D registration with downsampling

Control Point Pyramid (increase control points):
- multiscale_register_2d: Full-resolution 2D with coarse-to-fine control points
- multiscale_register_3d: Full-resolution 3D with coarse-to-fine control points

Transform Helpers:
- apply_transforms_2d: Apply sequence of 2D displacement fields
- apply_transforms_3d: Apply sequence of 3D displacement fields

Loss Functions
--------------
- CorrelationLoss: Negative Pearson correlation (default, same-modality)
- MAELoss: Mean Absolute Error
- MutualInformationLoss: Mutual Information with soft histogram (multi-modal)
- MonaiGlobalMILoss: MONAI's optimized Global MI (requires MONAI, faster)
- MonaiLocalNCCLoss: MONAI's Local Normalized Cross-Correlation (requires MONAI)

Regularization Weights
----------------------
The registration functions accept two key regularization weights:

smooth_weight : float
    Weight for displacement magnitude regularization. Penalizes large
    displacements to prevent unrealistic deformations. Higher values
    create more rigid transformations.
    - Default 2D: 0.005
    - Default 3D: 0.01
    - Range: [0, 1], typical values 0.001-0.1

bending_weight : float
    Weight for bending energy (second derivative) regularization.
    Penalizes high-frequency deformations to enforce smoothness.
    Higher values create smoother, more physically plausible deformations.
    - Default 2D: 0.01
    - Default 3D: 0.01
    - Range: [0, 1], typical values 0.001-0.1

Example:
    # For more rigid registration (less deformation)
    result = register_ffd_2d(moving, fixed, smooth_weight=0.1, bending_weight=0.1)
    
    # For more flexible registration (larger deformations allowed)
    result = register_ffd_2d(moving, fixed, smooth_weight=0.001, bending_weight=0.005)

Utilities
---------
- create_2d_grid_image: Visualize 2D control grid
- create_3d_grid_image: Visualize 3D control grid
- shepp_logan_2d: Generate 2D Shepp-Logan phantom
- shepp_logan_3d: Generate 3D Shepp-Logan phantom
- compare_loss_functions: Compare all loss functions with different approaches
"""

__version__ = "0.1.0"

# Core registration functions
from .ffd_2d import register_ffd_2d, apply_ffd_2d, create_2d_grid_image
from .ffd_3d import register_ffd_3d, apply_ffd_3d, create_3d_grid_image

# Pyramid registration
from .pyramid import (
    pyramid_register_2d,
    pyramid_register_3d,
    multiscale_register_2d,
    multiscale_register_3d,
    apply_transforms_2d,
    apply_transforms_3d,
)

# Loss functions
from .losses import (
    BaseLoss,
    CorrelationLoss,
    MAELoss,
    MutualInformationLoss,
    check_monai_available,
    get_loss_function,
)

# Conditionally import MONAI losses
try:
    from .losses import MonaiGlobalMILoss, MonaiLocalNCCLoss
    _MONAI_LOSSES = ["MonaiGlobalMILoss", "MonaiLocalNCCLoss"]
except ImportError:
    _MONAI_LOSSES = []

# Utilities
from .ffd_common import get_default_device, normalize_image
from .phantoms import shepp_logan_2d, shepp_logan_3d, apply_random_deformation
from .comparison import compare_loss_functions, get_available_loss_functions, RegistrationResult

__all__ = [
    # Version
    "__version__",
    # Core 2D
    "register_ffd_2d",
    "apply_ffd_2d",
    "create_2d_grid_image",
    # Core 3D
    "register_ffd_3d",
    "apply_ffd_3d",
    "create_3d_grid_image",
    # Image Pyramid (downsample images)
    "pyramid_register_2d",
    "pyramid_register_3d",
    # Control Point Pyramid (multiscale - full resolution)
    "multiscale_register_2d",
    "multiscale_register_3d",
    # Transform application helpers
    "apply_transforms_2d",
    "apply_transforms_3d",
    # Loss functions
    "BaseLoss",
    "CorrelationLoss",
    "MAELoss",
    "MutualInformationLoss",
    "check_monai_available",
    "get_loss_function",
    *_MONAI_LOSSES,  # MonaiGlobalMILoss if available
    # Utilities
    "get_default_device",
    "normalize_image",
    "shepp_logan_2d",
    "shepp_logan_3d",
    "apply_random_deformation",
    # Comparison utilities
    "compare_loss_functions",
    "get_available_loss_functions",
    "RegistrationResult",
]

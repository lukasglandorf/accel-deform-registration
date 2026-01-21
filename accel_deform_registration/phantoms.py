# -*- coding: utf-8 -*-
"""
Phantom generators for testing registration algorithms.

This module provides functions to generate synthetic test images
including the Shepp-Logan phantom and utilities for applying
controlled deformations.

Functions
---------
- shepp_logan_2d: Generate 2D Shepp-Logan phantom
- shepp_logan_3d: Generate 3D Shepp-Logan phantom
- apply_random_deformation: Apply random smooth deformation to an image/volume
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray


def _ellipse_2d(
    shape: Tuple[int, int],
    center: Tuple[float, float],
    axes: Tuple[float, float],
    angle: float,
    intensity: float,
) -> NDArray[np.float32]:
    """
    Generate a 2D ellipse.
    
    Parameters
    ----------
    shape : tuple
        Image shape (Y, X).
    center : tuple
        Ellipse center (cy, cx) as fractions of image size [0, 1].
    axes : tuple
        Semi-axes (ay, ax) as fractions of image size.
    angle : float
        Rotation angle in degrees.
    intensity : float
        Intensity value of the ellipse.
    
    Returns
    -------
    image : ndarray
        Image with the ellipse.
    """
    Y, X = shape
    
    # Convert to pixel coordinates
    cy, cx = center[0] * Y, center[1] * X
    ay, ax = axes[0] * Y, axes[1] * X
    
    # Create coordinate grids
    yy, xx = np.ogrid[:Y, :X]
    
    # Rotate coordinates
    theta = np.radians(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    
    x_rot = cos_t * (xx - cx) + sin_t * (yy - cy)
    y_rot = -sin_t * (xx - cx) + cos_t * (yy - cy)
    
    # Ellipse equation
    ellipse = (x_rot / ax) ** 2 + (y_rot / ay) ** 2 <= 1.0
    
    return (ellipse * intensity).astype(np.float32)


def shepp_logan_2d(
    size: int = 256,
    modified: bool = True,
) -> NDArray[np.float32]:
    """
    Generate a 2D Shepp-Logan phantom.
    
    The Shepp-Logan phantom is a standard test image used in medical
    imaging research, consisting of several ellipses of varying intensity.
    
    Parameters
    ----------
    size : int
        Image size (will be size x size). Default 256.
    modified : bool
        If True, use modified intensities for better visualization.
        Default True.
    
    Returns
    -------
    phantom : ndarray
        Phantom image of shape (size, size) with values in [0, 1].
    
    Examples
    --------
    >>> phantom = shepp_logan_2d(256)
    >>> phantom.shape
    (256, 256)
    """
    shape = (size, size)
    phantom = np.zeros(shape, dtype=np.float32)
    
    # Shepp-Logan parameters: (center_y, center_x), (axis_y, axis_x), angle, intensity
    # Parameters normalized to [0, 1] coordinate system (0.5 = center)
    if modified:
        # Modified Shepp-Logan for better contrast
        ellipses = [
            # (center_y, center_x), (axis_y, axis_x), angle, intensity
            ((0.5, 0.5), (0.345, 0.46), 0, 1.0),          # Outer ellipse
            ((0.5, 0.5), (0.3225, 0.4375), 0, -0.8),      # Inner dark ellipse
            ((0.44, 0.5), (0.125, 0.11), -18, -0.2),      # Right kidney
            ((0.56, 0.5), (0.125, 0.11), 18, -0.2),       # Left kidney
            ((0.5, 0.5), (0.125, 0.1875), 0, 0.15),       # Central vertical
            ((0.5, 0.5), (0.025, 0.023), 0, 0.15),        # Small center
            ((0.4, 0.5), (0.023, 0.023), 0, 0.15),        # Small right
            ((0.6, 0.5), (0.023, 0.023), 0, 0.15),        # Small left
            ((0.35, 0.6), (0.012, 0.023), 0, 0.15),       # Tiny 1
            ((0.65, 0.6), (0.012, 0.023), 0, 0.15),       # Tiny 2
        ]
    else:
        # Original Shepp-Logan intensities
        ellipses = [
            ((0.5, 0.5), (0.345, 0.46), 0, 2.0),
            ((0.5, 0.5), (0.3225, 0.4375), 0, -0.98),
            ((0.44, 0.5), (0.125, 0.11), -18, -0.02),
            ((0.56, 0.5), (0.125, 0.11), 18, -0.02),
            ((0.5, 0.5), (0.125, 0.1875), 0, 0.01),
            ((0.5, 0.5), (0.025, 0.023), 0, 0.01),
            ((0.4, 0.5), (0.023, 0.023), 0, 0.01),
            ((0.6, 0.5), (0.023, 0.023), 0, 0.01),
            ((0.35, 0.6), (0.012, 0.023), 0, 0.01),
            ((0.65, 0.6), (0.012, 0.023), 0, 0.01),
        ]
    
    for center, axes, angle, intensity in ellipses:
        phantom += _ellipse_2d(shape, center, axes, angle, intensity)
    
    # Normalize to [0, 1]
    phantom = np.clip(phantom, 0, None)
    if phantom.max() > 0:
        phantom = phantom / phantom.max()
    
    return phantom


def _ellipsoid_3d(
    shape: Tuple[int, int, int],
    center: Tuple[float, float, float],
    axes: Tuple[float, float, float],
    angles: Tuple[float, float, float],
    intensity: float,
) -> NDArray[np.float32]:
    """
    Generate a 3D ellipsoid.
    
    Parameters
    ----------
    shape : tuple
        Volume shape (Z, Y, X).
    center : tuple
        Ellipsoid center (cz, cy, cx) as fractions of volume size [0, 1].
    axes : tuple
        Semi-axes (az, ay, ax) as fractions of volume size.
    angles : tuple
        Rotation angles (not fully implemented, uses simple axis-aligned).
    intensity : float
        Intensity value of the ellipsoid.
    
    Returns
    -------
    volume : ndarray
        Volume with the ellipsoid.
    """
    Z, Y, X = shape
    
    # Convert to voxel coordinates
    cz, cy, cx = center[0] * Z, center[1] * Y, center[2] * X
    az, ay, ax = max(axes[0] * Z, 0.5), max(axes[1] * Y, 0.5), max(axes[2] * X, 0.5)
    
    # Create coordinate grids
    zz, yy, xx = np.ogrid[:Z, :Y, :X]
    
    # Ellipsoid equation (axis-aligned for simplicity)
    ellipsoid = (
        ((xx - cx) / ax) ** 2 +
        ((yy - cy) / ay) ** 2 +
        ((zz - cz) / az) ** 2
    ) <= 1.0
    
    return (ellipsoid * intensity).astype(np.float32)


def shepp_logan_3d(
    size: int = 128,
    modified: bool = True,
) -> NDArray[np.float32]:
    """
    Generate a 3D Shepp-Logan phantom.
    
    A 3D extension of the classic Shepp-Logan phantom, consisting
    of several ellipsoids of varying intensity.
    
    Parameters
    ----------
    size : int
        Volume size (will be size x size x size). Default 128.
    modified : bool
        If True, use modified intensities for better visualization.
        Default True.
    
    Returns
    -------
    phantom : ndarray
        Phantom volume of shape (size, size, size) with values in [0, 1].
    
    Examples
    --------
    >>> phantom = shepp_logan_3d(128)
    >>> phantom.shape
    (128, 128, 128)
    """
    shape = (size, size, size)
    phantom = np.zeros(shape, dtype=np.float32)
    
    # 3D Shepp-Logan parameters: (center_z, center_y, center_x), (axis_z, axis_y, axis_x), angles, intensity
    # Parameters normalized to [0, 1] coordinate system (0.5 = center)
    if modified:
        ellipsoids = [
            # Outer skull
            ((0.5, 0.5, 0.5), (0.35, 0.345, 0.46), (0, 0, 0), 1.0),
            # Inner (brain matter)
            ((0.5, 0.5, 0.5), (0.32, 0.3225, 0.4375), (0, 0, 0), -0.8),
            # Right ventricle
            ((0.5, 0.44, 0.5), (0.15, 0.125, 0.11), (0, 0, 0), -0.2),
            # Left ventricle
            ((0.5, 0.56, 0.5), (0.15, 0.125, 0.11), (0, 0, 0), -0.2),
            # Central structure
            ((0.5, 0.5, 0.5), (0.2, 0.125, 0.1875), (0, 0, 0), 0.15),
            # Small central nucleus
            ((0.5, 0.5, 0.5), (0.05, 0.025, 0.023), (0, 0, 0), 0.15),
            # Additional features
            ((0.5, 0.4, 0.5), (0.04, 0.023, 0.023), (0, 0, 0), 0.1),
            ((0.5, 0.6, 0.5), (0.04, 0.023, 0.023), (0, 0, 0), 0.1),
            ((0.5, 0.35, 0.6), (0.03, 0.012, 0.023), (0, 0, 0), 0.1),
            ((0.5, 0.65, 0.6), (0.03, 0.012, 0.023), (0, 0, 0), 0.1),
        ]
    else:
        ellipsoids = [
            ((0.5, 0.5, 0.5), (0.35, 0.345, 0.46), (0, 0, 0), 2.0),
            ((0.5, 0.5, 0.5), (0.32, 0.3225, 0.4375), (0, 0, 0), -0.98),
            ((0.5, 0.44, 0.5), (0.15, 0.125, 0.11), (0, 0, 0), -0.02),
            ((0.5, 0.56, 0.5), (0.15, 0.125, 0.11), (0, 0, 0), -0.02),
            ((0.5, 0.5, 0.5), (0.2, 0.125, 0.1875), (0, 0, 0), 0.01),
            ((0.5, 0.5, 0.5), (0.05, 0.025, 0.023), (0, 0, 0), 0.01),
            ((0.5, 0.4, 0.5), (0.04, 0.023, 0.023), (0, 0, 0), 0.01),
            ((0.5, 0.6, 0.5), (0.04, 0.023, 0.023), (0, 0, 0), 0.01),
            ((0.5, 0.35, 0.6), (0.03, 0.012, 0.023), (0, 0, 0), 0.01),
            ((0.5, 0.65, 0.6), (0.03, 0.012, 0.023), (0, 0, 0), 0.01),
        ]
    
    for center, axes, angles, intensity in ellipsoids:
        phantom += _ellipsoid_3d(shape, center, axes, angles, intensity)
    
    # Normalize to [0, 1]
    phantom = np.clip(phantom, 0, None)
    if phantom.max() > 0:
        phantom = phantom / phantom.max()
    
    return phantom


def apply_random_deformation(
    image: NDArray[np.floating],
    max_displacement: float = 10.0,
    n_control_points: int = 5,
    seed: Optional[int] = None,
    smooth_sigma: Optional[float] = None,
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Apply a random smooth deformation to an image or volume.
    
    Uses B-spline-like smooth random displacement fields.
    
    Parameters
    ----------
    image : ndarray
        Input image (2D) or volume (3D).
    max_displacement : float
        Maximum displacement in pixels/voxels. Default 10.0.
    n_control_points : int
        Number of control points along each axis for generating
        the random displacement field. Default 5.
    seed : int, optional
        Random seed for reproducibility.
    smooth_sigma : float, optional
        Gaussian smoothing sigma for the displacement field.
        If None, computed automatically from image size.
    
    Returns
    -------
    deformed : ndarray
        Deformed image/volume.
    displacement : ndarray
        Displacement field used, shape (..., ndim).
    
    Examples
    --------
    >>> phantom = shepp_logan_2d(256)
    >>> deformed, displacement = apply_random_deformation(phantom, max_displacement=15)
    """
    from scipy import ndimage
    
    if seed is not None:
        np.random.seed(seed)
    
    ndim = image.ndim
    shape = image.shape
    
    # Auto-compute smoothing sigma if not provided
    if smooth_sigma is None:
        smooth_sigma = min(shape) / (n_control_points * 2)
    
    # Generate random control point displacements
    ctrl_shape = tuple([n_control_points] * ndim)
    
    displacement_field = np.zeros(shape + (ndim,), dtype=np.float32)
    
    for dim in range(ndim):
        # Random displacements at control points
        ctrl_disps = np.random.uniform(-1, 1, ctrl_shape).astype(np.float32)
        
        # Upsample to full resolution using zoom
        zoom_factors = tuple(s / n for s, n in zip(shape, ctrl_shape))
        disp_component = ndimage.zoom(ctrl_disps, zoom_factors, order=3)
        
        # Resize to exact shape (in case of rounding)
        if disp_component.shape != shape:
            slices = tuple(slice(0, s) for s in shape)
            disp_component = disp_component[slices]
        
        # Apply Gaussian smoothing for smoothness
        disp_component = ndimage.gaussian_filter(disp_component, sigma=smooth_sigma)
        
        # Scale to desired max displacement
        disp_component = disp_component / (np.abs(disp_component).max() + 1e-8) * max_displacement
        
        displacement_field[..., dim] = disp_component
    
    # Apply the deformation
    if ndim == 2:
        deformed = _apply_displacement_2d(image, displacement_field)
    else:
        deformed = _apply_displacement_3d(image, displacement_field)
    
    return deformed.astype(np.float32), displacement_field


def _apply_displacement_2d(
    image: NDArray[np.floating],
    displacement: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Apply a 2D displacement field using scipy map_coordinates."""
    from scipy import ndimage
    
    Y, X = image.shape
    
    # Create coordinate grids
    yy, xx = np.meshgrid(np.arange(Y), np.arange(X), indexing='ij')
    
    # Compute source coordinates
    source_x = xx - displacement[:, :, 0]
    source_y = yy - displacement[:, :, 1]
    
    # Interpolate
    deformed = ndimage.map_coordinates(
        image.astype(np.float32),
        [source_y, source_x],
        order=1,
        mode='nearest',
    )
    
    return deformed


def _apply_displacement_3d(
    volume: NDArray[np.floating],
    displacement: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Apply a 3D displacement field using scipy map_coordinates."""
    from scipy import ndimage
    
    Z, Y, X = volume.shape
    
    # Create coordinate grids
    zz, yy, xx = np.meshgrid(
        np.arange(Z), np.arange(Y), np.arange(X), indexing='ij'
    )
    
    # Compute source coordinates
    source_x = xx - displacement[:, :, :, 0]
    source_y = yy - displacement[:, :, :, 1]
    source_z = zz - displacement[:, :, :, 2]
    
    # Interpolate
    deformed = ndimage.map_coordinates(
        volume.astype(np.float32),
        [source_z, source_y, source_x],
        order=1,
        mode='nearest',
    )
    
    return deformed


def create_checkerboard_2d(
    size: int = 256,
    n_squares: int = 8,
) -> NDArray[np.float32]:
    """
    Create a 2D checkerboard pattern (useful for visualizing deformations).
    
    Parameters
    ----------
    size : int
        Image size. Default 256.
    n_squares : int
        Number of squares along each axis. Default 8.
    
    Returns
    -------
    checkerboard : ndarray
        Checkerboard image with values 0 and 1.
    """
    square_size = size // n_squares
    checkerboard = np.zeros((size, size), dtype=np.float32)
    
    for i in range(n_squares):
        for j in range(n_squares):
            if (i + j) % 2 == 0:
                y_start, y_end = i * square_size, (i + 1) * square_size
                x_start, x_end = j * square_size, (j + 1) * square_size
                checkerboard[y_start:y_end, x_start:x_end] = 1.0
    
    return checkerboard


def create_checkerboard_3d(
    size: int = 128,
    n_cubes: int = 4,
) -> NDArray[np.float32]:
    """
    Create a 3D checkerboard pattern.
    
    Parameters
    ----------
    size : int
        Volume size. Default 128.
    n_cubes : int
        Number of cubes along each axis. Default 4.
    
    Returns
    -------
    checkerboard : ndarray
        Checkerboard volume with values 0 and 1.
    """
    cube_size = size // n_cubes
    checkerboard = np.zeros((size, size, size), dtype=np.float32)
    
    for i in range(n_cubes):
        for j in range(n_cubes):
            for k in range(n_cubes):
                if (i + j + k) % 2 == 0:
                    z_start, z_end = i * cube_size, (i + 1) * cube_size
                    y_start, y_end = j * cube_size, (j + 1) * cube_size
                    x_start, x_end = k * cube_size, (k + 1) * cube_size
                    checkerboard[z_start:z_end, y_start:y_end, x_start:x_end] = 1.0
    
    return checkerboard

# -*- coding: utf-8 -*-
"""
Pyramid (multi-resolution) registration wrappers.

This module provides coarse-to-fine registration by running FFD
at multiple resolution levels.

Two Approaches
--------------
1. **Image Pyramid** (pyramid_register_2d/3d):
   - Downsample images at each level
   - Traditional approach, good for large deformations
   - Displacement fields are composed into a single field

2. **Control Point Pyramid** (multiscale_register_2d/3d):
   - Work at full image resolution throughout
   - Increase number of control points at each level
   - Returns list of transforms to apply sequentially
   - Better for capturing fine details with fast GPU computation

Functions
---------
Image Pyramid:
- pyramid_register_2d: Multi-resolution 2D FFD registration
- pyramid_register_3d: Multi-resolution 3D FFD registration

Control Point Pyramid:
- multiscale_register_2d: Multi-scale 2D FFD with increasing control points
- multiscale_register_3d: Multi-scale 3D FFD with increasing control points

Transform Application:
- apply_transforms_2d: Apply a sequence of 2D displacement fields
- apply_transforms_3d: Apply a sequence of 3D displacement fields
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from typing import Dict, Any, Optional, List, Tuple, Union
from numpy.typing import NDArray

from .ffd_2d import register_ffd_2d, apply_ffd_2d
from .ffd_3d import register_ffd_3d, apply_ffd_3d
from .losses import BaseLoss


def _downsample_2d(
    image: NDArray[np.floating],
    factor: int,
) -> NDArray[np.float32]:
    """Downsample a 2D image by the given factor."""
    if factor == 1:
        return image.astype(np.float32)
    
    # Apply Gaussian smoothing to avoid aliasing
    sigma = factor / 2.0
    smoothed = ndimage.gaussian_filter(image.astype(np.float32), sigma=sigma)
    
    # Downsample
    return smoothed[::factor, ::factor]


def _downsample_3d(
    volume: NDArray[np.floating],
    factor: int,
) -> NDArray[np.float32]:
    """Downsample a 3D volume by the given factor."""
    if factor == 1:
        return volume.astype(np.float32)
    
    # Apply Gaussian smoothing to avoid aliasing
    sigma = factor / 2.0
    smoothed = ndimage.gaussian_filter(volume.astype(np.float32), sigma=sigma)
    
    # Downsample
    return smoothed[::factor, ::factor, ::factor]


def _upsample_displacement_2d(
    displacement: NDArray[np.floating],
    target_shape: Tuple[int, int],
    scale_factor: float,
) -> NDArray[np.float32]:
    """
    Upsample a 2D displacement field and scale displacement values.
    
    Parameters
    ----------
    displacement : ndarray
        Displacement field of shape (Y, X, 2).
    target_shape : tuple
        Target shape (Y, X).
    scale_factor : float
        Factor to scale displacement values (ratio of target to source size).
    
    Returns
    -------
    upsampled : ndarray
        Upsampled displacement field of shape (target_Y, target_X, 2).
    """
    Y, X = target_shape
    
    # Interpolate each displacement component
    upsampled = np.zeros((Y, X, 2), dtype=np.float32)
    
    for i in range(2):
        upsampled[:, :, i] = ndimage.zoom(
            displacement[:, :, i],
            (Y / displacement.shape[0], X / displacement.shape[1]),
            order=1,  # Bilinear
        ) * scale_factor
    
    return upsampled


def _upsample_displacement_3d(
    displacement: NDArray[np.floating],
    target_shape: Tuple[int, int, int],
    scale_factor: float,
) -> NDArray[np.float32]:
    """
    Upsample a 3D displacement field and scale displacement values.
    
    Parameters
    ----------
    displacement : ndarray
        Displacement field of shape (Z, Y, X, 3).
    target_shape : tuple
        Target shape (Z, Y, X).
    scale_factor : float
        Factor to scale displacement values.
    
    Returns
    -------
    upsampled : ndarray
        Upsampled displacement field of shape (target_Z, target_Y, target_X, 3).
    """
    Z, Y, X = target_shape
    
    # Interpolate each displacement component
    upsampled = np.zeros((Z, Y, X, 3), dtype=np.float32)
    
    zoom_factors = (
        Z / displacement.shape[0],
        Y / displacement.shape[1],
        X / displacement.shape[2],
    )
    
    for i in range(3):
        upsampled[:, :, :, i] = ndimage.zoom(
            displacement[:, :, :, i],
            zoom_factors,
            order=1,  # Trilinear
        ) * scale_factor
    
    return upsampled


def pyramid_register_2d(
    moving: NDArray[np.floating],
    fixed: NDArray[np.floating],
    levels: List[int] = [4, 2, 1],
    grid_spacings: Optional[List[int]] = None,
    iterations_per_level: Optional[List[int]] = None,
    smooth_weight: float = 0.005,
    bending_weight: float = 0.01,
    lr: float = 0.5,
    loss_fn: Optional[BaseLoss] = None,
    device=None,
    verbose: bool = True,
) -> Tuple[NDArray[np.float32], Dict[str, Any]]:
    """
    Multi-resolution (pyramid) 2D FFD registration.
    
    Performs coarse-to-fine registration by starting at low resolution
    and progressively refining at higher resolutions.
    
    Parameters
    ----------
    moving : ndarray
        Moving (source) image to be warped, shape (Y, X).
    fixed : ndarray
        Fixed (target) image, shape (Y, X).
    levels : list of int
        Downsample factors for each level, from coarse to fine.
        Default [4, 2, 1] means 4x, 2x, then full resolution.
    grid_spacings : list of int, optional
        Grid spacing for each level. If None, uses [50, 25, 12] or
        auto-computed based on levels.
    iterations_per_level : list of int, optional
        Number of iterations per level. If None, uses [500, 300, 200].
    smooth_weight : float
        Smoothness regularization weight. Default 0.005.
    bending_weight : float
        Bending energy weight. Default 0.01.
    lr : float
        Learning rate. Default 0.5.
    loss_fn : BaseLoss, optional
        Loss function. If None, uses CorrelationLoss.
    device : torch.device, optional
        PyTorch device. If None, auto-detect.
    verbose : bool
        Print progress. Default True.
    
    Returns
    -------
    displacement : ndarray
        Final displacement field of shape (Y, X, 2) in pixel units.
    info : dict
        Registration metadata including:
        - 'levels': List of downsample factors used
        - 'level_info': List of info dicts from each level
        - 'total_iterations': Total iterations across all levels
    
    Examples
    --------
    >>> disp, info = pyramid_register_2d(
    ...     moving, fixed,
    ...     levels=[4, 2, 1],
    ...     grid_spacings=[50, 25, 12],
    ... )
    """
    # Default parameters
    if grid_spacings is None:
        grid_spacings = [max(25, 100 // level) for level in levels]
    if iterations_per_level is None:
        iterations_per_level = [500, 300, 200][:len(levels)]
    
    # Validate lengths
    n_levels = len(levels)
    if len(grid_spacings) != n_levels:
        raise ValueError(f"grid_spacings length ({len(grid_spacings)}) must match levels ({n_levels})")
    if len(iterations_per_level) != n_levels:
        raise ValueError(f"iterations_per_level length ({len(iterations_per_level)}) must match levels ({n_levels})")
    
    Y_full, X_full = moving.shape
    
    # Initialize displacement field
    displacement = np.zeros((Y_full, X_full, 2), dtype=np.float32)
    level_infos = []
    total_iterations = 0
    
    if verbose:
        print(f"Pyramid 2D registration with {n_levels} levels: {levels}")
    
    for level_idx, (downsample, grid_sp, n_iter) in enumerate(
        zip(levels, grid_spacings, iterations_per_level)
    ):
        if verbose:
            print(f"\n--- Level {level_idx + 1}/{n_levels} (downsample={downsample}x) ---")
        
        # Downsample images
        moving_ds = _downsample_2d(moving, downsample)
        fixed_ds = _downsample_2d(fixed, downsample)
        
        # Downsample and scale current displacement
        if downsample > 1:
            disp_ds = _upsample_displacement_2d(
                displacement,
                moving_ds.shape,
                scale_factor=1.0 / downsample,
            )
        else:
            disp_ds = displacement.copy()
        
        # Apply current displacement to moving image
        if np.abs(disp_ds).max() > 0.1:
            moving_warped = apply_ffd_2d(moving_ds, disp_ds, device=device)
        else:
            moving_warped = moving_ds
        
        # Register at this level
        disp_level, _, _, _, info = register_ffd_2d(
            moving_warped, fixed_ds,
            grid_spacing=grid_sp,
            smooth_weight=smooth_weight,
            bending_weight=bending_weight,
            n_iterations=n_iter,
            lr=lr,
            loss_fn=loss_fn,
            device=device,
            verbose=verbose,
        )
        
        level_infos.append(info)
        total_iterations += info['iterations']
        
        # Compose displacements
        # Upsample level displacement to full resolution
        if downsample > 1:
            disp_level_full = _upsample_displacement_2d(
                disp_level,
                (Y_full, X_full),
                scale_factor=float(downsample),
            )
        else:
            disp_level_full = disp_level
        
        # Add to cumulative displacement
        displacement = displacement + disp_level_full
    
    # Final info
    info = {
        'levels': levels,
        'grid_spacings': grid_spacings,
        'iterations_per_level': iterations_per_level,
        'level_info': level_infos,
        'total_iterations': total_iterations,
        'max_displacement': float(np.linalg.norm(displacement.reshape(-1, 2), axis=1).max()),
    }
    
    if verbose:
        print(f"\nPyramid complete. Total iterations: {total_iterations}")
        print(f"Max displacement: {info['max_displacement']:.1f} pixels")
    
    return displacement, info


def pyramid_register_3d(
    moving: NDArray[np.floating],
    fixed: NDArray[np.floating],
    levels: List[int] = [4, 2, 1],
    grid_spacings: Optional[List[int]] = None,
    iterations_per_level: Optional[List[int]] = None,
    smooth_weight: float = 0.01,
    bending_weight: float = 0.01,
    lr: float = 0.5,
    mask: Optional[NDArray[np.floating]] = None,
    loss_fn: Optional[BaseLoss] = None,
    device=None,
    verbose: bool = True,
) -> Tuple[NDArray[np.float32], Dict[str, Any]]:
    """
    Multi-resolution (pyramid) 3D FFD registration.
    
    Performs coarse-to-fine registration for 3D volumes.
    
    Parameters
    ----------
    moving : ndarray
        Moving (source) volume to be warped, shape (Z, Y, X).
    fixed : ndarray
        Fixed (target) volume, shape (Z, Y, X).
    levels : list of int
        Downsample factors for each level. Default [4, 2, 1].
    grid_spacings : list of int, optional
        Grid spacing for each level. If None, auto-computed.
    iterations_per_level : list of int, optional
        Number of iterations per level. If None, uses [200, 150, 100].
    smooth_weight : float
        Smoothness regularization weight. Default 0.01.
    bending_weight : float
        Bending energy weight. Default 0.01.
    lr : float
        Learning rate. Default 0.5.
    mask : ndarray, optional
        Optional mask for weighted loss. Shape (Z, Y, X).
    loss_fn : BaseLoss, optional
        Loss function. If None, uses CorrelationLoss.
    device : torch.device, optional
        PyTorch device. If None, auto-detect.
    verbose : bool
        Print progress. Default True.
    
    Returns
    -------
    displacement : ndarray
        Final displacement field of shape (Z, Y, X, 3) in voxel units.
    info : dict
        Registration metadata.
    
    Examples
    --------
    >>> disp, info = pyramid_register_3d(
    ...     moving_vol, fixed_vol,
    ...     levels=[4, 2, 1],
    ...     grid_spacings=[40, 20, 10],
    ... )
    """
    # Default parameters
    if grid_spacings is None:
        grid_spacings = [max(20, 80 // level) for level in levels]
    if iterations_per_level is None:
        iterations_per_level = [200, 150, 100][:len(levels)]
    
    # Validate lengths
    n_levels = len(levels)
    if len(grid_spacings) != n_levels:
        raise ValueError(f"grid_spacings length ({len(grid_spacings)}) must match levels ({n_levels})")
    if len(iterations_per_level) != n_levels:
        raise ValueError(f"iterations_per_level length ({len(iterations_per_level)}) must match levels ({n_levels})")
    
    Z_full, Y_full, X_full = moving.shape
    
    # Initialize displacement field
    displacement = np.zeros((Z_full, Y_full, X_full, 3), dtype=np.float32)
    level_infos = []
    total_iterations = 0
    
    if verbose:
        print(f"Pyramid 3D registration with {n_levels} levels: {levels}")
    
    for level_idx, (downsample, grid_sp, n_iter) in enumerate(
        zip(levels, grid_spacings, iterations_per_level)
    ):
        if verbose:
            print(f"\n--- Level {level_idx + 1}/{n_levels} (downsample={downsample}x) ---")
        
        # Downsample volumes
        moving_ds = _downsample_3d(moving, downsample)
        fixed_ds = _downsample_3d(fixed, downsample)
        
        # Downsample mask if provided
        mask_ds = None
        if mask is not None:
            mask_ds = _downsample_3d(mask, downsample)
        
        # Downsample and scale current displacement
        if downsample > 1:
            disp_ds = _upsample_displacement_3d(
                displacement,
                moving_ds.shape,
                scale_factor=1.0 / downsample,
            )
        else:
            disp_ds = displacement.copy()
        
        # Apply current displacement to moving volume
        if np.abs(disp_ds).max() > 0.1:
            moving_warped = apply_ffd_3d(moving_ds, disp_ds, device=device)
        else:
            moving_warped = moving_ds
        
        # Register at this level
        disp_level, _, _, _, info = register_ffd_3d(
            moving_warped, fixed_ds,
            grid_spacing=grid_sp,
            smooth_weight=smooth_weight,
            bending_weight=bending_weight,
            n_iterations=n_iter,
            lr=lr,
            mask=mask_ds,
            loss_fn=loss_fn,
            device=device,
            verbose=verbose,
        )
        
        level_infos.append(info)
        total_iterations += info['iterations']
        
        # Compose displacements
        # Upsample level displacement to full resolution
        if downsample > 1:
            disp_level_full = _upsample_displacement_3d(
                disp_level,
                (Z_full, Y_full, X_full),
                scale_factor=float(downsample),
            )
        else:
            disp_level_full = disp_level
        
        # Add to cumulative displacement
        displacement = displacement + disp_level_full
    
    # Final info
    info = {
        'levels': levels,
        'grid_spacings': grid_spacings,
        'iterations_per_level': iterations_per_level,
        'level_info': level_infos,
        'total_iterations': total_iterations,
        'max_displacement': float(np.linalg.norm(displacement.reshape(-1, 3), axis=1).max()),
    }
    
    if verbose:
        print(f"\nPyramid complete. Total iterations: {total_iterations}")
        print(f"Max displacement: {info['max_displacement']:.1f} voxels")
    
    return displacement, info


# =============================================================================
# Control Point Pyramid (Multiscale) Registration
# =============================================================================

def _normalize_grid_spacing(
    grid_spacing: Union[int, Tuple[int, ...], List[int]],
    ndim: int,
) -> Tuple[int, ...]:
    """
    Normalize grid spacing to a tuple for all dimensions.
    
    Parameters
    ----------
    grid_spacing : int or tuple/list
        Spacing as a single int (same for all dims) or per-dimension.
    ndim : int
        Number of dimensions (2 or 3).
    
    Returns
    -------
    tuple
        Grid spacing for each dimension.
    """
    if isinstance(grid_spacing, (int, np.integer)):
        return tuple([int(grid_spacing)] * ndim)
    else:
        if len(grid_spacing) != ndim:
            raise ValueError(f"grid_spacing must have {ndim} elements, got {len(grid_spacing)}")
        return tuple(int(s) for s in grid_spacing)


def apply_transforms_2d(
    image: NDArray[np.floating],
    displacements: List[NDArray[np.float32]],
    device=None,
) -> NDArray[np.float32]:
    """
    Apply a sequence of 2D displacement fields to an image.
    
    Each displacement is applied in order, transforming the result of
    the previous transformation.
    
    Parameters
    ----------
    image : ndarray
        Input image of shape (Y, X).
    displacements : list of ndarray
        List of displacement fields, each of shape (Y, X, 2).
        Applied in order from first to last.
    device : torch.device, optional
        PyTorch device for computation.
    
    Returns
    -------
    result : ndarray
        Transformed image of shape (Y, X).
    
    Examples
    --------
    >>> # Apply coarse-to-fine transforms
    >>> result = apply_transforms_2d(image, [coarse_disp, fine_disp])
    """
    result = image.astype(np.float32)
    
    for disp in displacements:
        result = apply_ffd_2d(result, disp, device=device)
    
    return result


def apply_transforms_3d(
    volume: NDArray[np.floating],
    displacements: List[NDArray[np.float32]],
    device=None,
) -> NDArray[np.float32]:
    """
    Apply a sequence of 3D displacement fields to a volume.
    
    Each displacement is applied in order, transforming the result of
    the previous transformation.
    
    Parameters
    ----------
    volume : ndarray
        Input volume of shape (Z, Y, X).
    displacements : list of ndarray
        List of displacement fields, each of shape (Z, Y, X, 3).
        Applied in order from first to last.
    device : torch.device, optional
        PyTorch device for computation.
    
    Returns
    -------
    result : ndarray
        Transformed volume of shape (Z, Y, X).
    
    Examples
    --------
    >>> # Apply coarse-to-fine transforms
    >>> result = apply_transforms_3d(volume, [coarse_disp, fine_disp])
    """
    result = volume.astype(np.float32)
    
    for disp in displacements:
        result = apply_ffd_3d(result, disp, device=device)
    
    return result


def multiscale_register_2d(
    moving: NDArray[np.floating],
    fixed: NDArray[np.floating],
    grid_spacings: List[Union[int, Tuple[int, int]]] = [100, 50, 25],
    iterations_per_level: Optional[List[int]] = None,
    smooth_weights: Optional[Union[float, List[float]]] = None,
    bending_weights: Optional[Union[float, List[float]]] = None,
    lr: float = 0.5,
    loss_fn: Optional[BaseLoss] = None,
    device=None,
    verbose: bool = True,
) -> Tuple[List[NDArray[np.float32]], Dict[str, Any]]:
    """
    Multi-scale 2D FFD registration with increasing control points.
    
    Unlike pyramid_register_2d which downsamples images, this works at
    full resolution throughout but uses coarse-to-fine control point grids.
    Returns a list of displacement fields that should be applied sequentially.
    
    Parameters
    ----------
    moving : ndarray
        Moving (source) image to be warped, shape (Y, X).
    fixed : ndarray
        Fixed (target) image, shape (Y, X).
    grid_spacings : list of int or tuple
        Control point spacing for each level, from coarse to fine.
        Each element can be:
        - int: Same spacing for Y and X
        - tuple (spacing_y, spacing_x): Different spacing per dimension
        Default [100, 50, 25] means 3 levels with decreasing spacing.
    iterations_per_level : list of int, optional
        Number of iterations per level. If None, uses [1000, 500, 300].
    smooth_weights : float or list of float, optional
        Smoothness weight for each level. If a single float, uses same
        for all levels. If None, uses 0.005.
    bending_weights : float or list of float, optional
        Bending energy weight for each level. If a single float, uses same
        for all levels. If None, uses 0.01.
    lr : float
        Learning rate. Default 0.5.
    loss_fn : BaseLoss, optional
        Loss function. If None, uses CorrelationLoss.
    device : torch.device, optional
        PyTorch device. If None, auto-detect.
    verbose : bool
        Print progress. Default True.
    
    Returns
    -------
    displacements : list of ndarray
        List of displacement fields, one per level, each of shape (Y, X, 2).
        Apply sequentially using apply_transforms_2d().
    info : dict
        Registration metadata including:
        - 'grid_spacings': List of (spacing_y, spacing_x) tuples used
        - 'iterations_per_level': Iterations used at each level
        - 'smooth_weights': Smooth weight used at each level
        - 'bending_weights': Bending weight used at each level
        - 'level_info': List of info dicts from each level
        - 'total_iterations': Total iterations across all levels
        - 'max_displacements': Max displacement at each level
    
    Notes
    -----
    This approach is ideal when:
    - GPU computation is fast enough to work at full resolution
    - You want to preserve fine details from the start
    - The deformation is relatively local
    
    The transforms are not composed into a single field. Instead, use
    apply_transforms_2d() to apply them sequentially. This avoids
    interpolation artifacts from field composition.
    
    Examples
    --------
    >>> # Basic usage with default 3 levels
    >>> disps, info = multiscale_register_2d(moving, fixed)
    >>> warped = apply_transforms_2d(moving, disps)
    
    >>> # Custom per-level settings
    >>> disps, info = multiscale_register_2d(
    ...     moving, fixed,
    ...     grid_spacings=[(100, 80), (50, 40), (25, 20)],  # Y, X spacing
    ...     iterations_per_level=[800, 400, 200],
    ...     smooth_weights=[0.01, 0.005, 0.001],
    ...     bending_weights=[0.02, 0.01, 0.005],
    ... )
    """
    n_levels = len(grid_spacings)
    
    # Set defaults
    if iterations_per_level is None:
        iterations_per_level = [1000, 500, 300][:n_levels]
        if len(iterations_per_level) < n_levels:
            iterations_per_level.extend([200] * (n_levels - len(iterations_per_level)))
    
    # Normalize smooth_weights to list
    if smooth_weights is None:
        smooth_weights = [0.005] * n_levels
    elif isinstance(smooth_weights, (int, float)):
        smooth_weights = [float(smooth_weights)] * n_levels
    
    # Normalize bending_weights to list
    if bending_weights is None:
        bending_weights = [0.01] * n_levels
    elif isinstance(bending_weights, (int, float)):
        bending_weights = [float(bending_weights)] * n_levels
    
    # Normalize grid_spacings to list of tuples
    normalized_spacings = [_normalize_grid_spacing(gs, 2) for gs in grid_spacings]
    
    # Validate lengths
    if len(iterations_per_level) != n_levels:
        raise ValueError(f"iterations_per_level length ({len(iterations_per_level)}) must match grid_spacings ({n_levels})")
    if len(smooth_weights) != n_levels:
        raise ValueError(f"smooth_weights length ({len(smooth_weights)}) must match grid_spacings ({n_levels})")
    if len(bending_weights) != n_levels:
        raise ValueError(f"bending_weights length ({len(bending_weights)}) must match grid_spacings ({n_levels})")
    
    displacements = []
    level_infos = []
    total_iterations = 0
    max_displacements = []
    
    if verbose:
        print(f"Multiscale 2D registration with {n_levels} levels")
        print(f"Grid spacings (Y, X): {normalized_spacings}")
    
    # Start with moving image
    current_moving = moving.astype(np.float32)
    
    for level_idx in range(n_levels):
        spacing = normalized_spacings[level_idx]
        n_iter = iterations_per_level[level_idx]
        smooth_w = smooth_weights[level_idx]
        bending_w = bending_weights[level_idx]
        
        if verbose:
            print(f"\n--- Level {level_idx + 1}/{n_levels} (spacing={spacing}, "
                  f"smooth={smooth_w:.4f}, bending={bending_w:.4f}) ---")
        
        # Register at this level using per-axis spacing
        disp, _, _, _, info = register_ffd_2d(
            current_moving, fixed,
            grid_spacing=spacing,
            smooth_weight=smooth_w,
            bending_weight=bending_w,
            n_iterations=n_iter,
            lr=lr,
            loss_fn=loss_fn,
            device=device,
            verbose=verbose,
        )
        
        displacements.append(disp)
        level_infos.append(info)
        total_iterations += info['iterations']
        max_displacements.append(info['max_displacement'])
        
        # Apply this displacement to get the new moving image for next level
        current_moving = apply_ffd_2d(current_moving, disp, device=device)
    
    # Final info
    result_info = {
        'grid_spacings': normalized_spacings,
        'iterations_per_level': iterations_per_level,
        'smooth_weights': smooth_weights,
        'bending_weights': bending_weights,
        'level_info': level_infos,
        'total_iterations': total_iterations,
        'max_displacements': max_displacements,
    }
    
    if verbose:
        print(f"\nMultiscale complete. Total iterations: {total_iterations}")
        print(f"Max displacement per level: {[f'{d:.1f}' for d in max_displacements]}")
    
    return displacements, result_info


def multiscale_register_3d(
    moving: NDArray[np.floating],
    fixed: NDArray[np.floating],
    grid_spacings: List[Union[int, Tuple[int, int, int]]] = [80, 40, 20],
    iterations_per_level: Optional[List[int]] = None,
    smooth_weights: Optional[Union[float, List[float]]] = None,
    bending_weights: Optional[Union[float, List[float]]] = None,
    lr: float = 0.5,
    mask: Optional[NDArray[np.floating]] = None,
    loss_fn: Optional[BaseLoss] = None,
    device=None,
    verbose: bool = True,
) -> Tuple[List[NDArray[np.float32]], Dict[str, Any]]:
    """
    Multi-scale 3D FFD registration with increasing control points.
    
    Unlike pyramid_register_3d which downsamples volumes, this works at
    full resolution throughout but uses coarse-to-fine control point grids.
    Returns a list of displacement fields that should be applied sequentially.
    
    Parameters
    ----------
    moving : ndarray
        Moving (source) volume to be warped, shape (Z, Y, X).
    fixed : ndarray
        Fixed (target) volume, shape (Z, Y, X).
    grid_spacings : list of int or tuple
        Control point spacing for each level, from coarse to fine.
        Each element can be:
        - int: Same spacing for Z, Y, and X
        - tuple (spacing_z, spacing_y, spacing_x): Different spacing per dimension
        Default [80, 40, 20] means 3 levels with decreasing spacing.
    iterations_per_level : list of int, optional
        Number of iterations per level. If None, uses [300, 200, 100].
    smooth_weights : float or list of float, optional
        Smoothness weight for each level. If a single float, uses same
        for all levels. If None, uses 0.01.
    bending_weights : float or list of float, optional
        Bending energy weight for each level. If a single float, uses same
        for all levels. If None, uses 0.01.
    lr : float
        Learning rate. Default 0.5.
    mask : ndarray, optional
        Optional mask of shape (Z, Y, X) for weighted loss.
    loss_fn : BaseLoss, optional
        Loss function. If None, uses CorrelationLoss.
    device : torch.device, optional
        PyTorch device. If None, auto-detect.
    verbose : bool
        Print progress. Default True.
    
    Returns
    -------
    displacements : list of ndarray
        List of displacement fields, one per level, each of shape (Z, Y, X, 3).
        Apply sequentially using apply_transforms_3d().
    info : dict
        Registration metadata including:
        - 'grid_spacings': List of (spacing_z, spacing_y, spacing_x) tuples used
        - 'iterations_per_level': Iterations used at each level
        - 'smooth_weights': Smooth weight used at each level
        - 'bending_weights': Bending weight used at each level
        - 'level_info': List of info dicts from each level
        - 'total_iterations': Total iterations across all levels
        - 'max_displacements': Max displacement at each level
    
    Notes
    -----
    This approach is ideal when:
    - GPU computation is fast enough to work at full resolution
    - You want to preserve fine details from the start
    - The deformation is relatively local
    
    The transforms are not composed into a single field. Instead, use
    apply_transforms_3d() to apply them sequentially. This avoids
    interpolation artifacts from field composition.
    
    Examples
    --------
    >>> # Basic usage with default 3 levels
    >>> disps, info = multiscale_register_3d(moving, fixed)
    >>> warped = apply_transforms_3d(moving, disps)
    
    >>> # Custom per-level settings
    >>> disps, info = multiscale_register_3d(
    ...     moving, fixed,
    ...     grid_spacings=[(80, 60, 60), (40, 30, 30), (20, 15, 15)],  # Z, Y, X
    ...     iterations_per_level=[250, 150, 75],
    ...     smooth_weights=[0.02, 0.01, 0.005],
    ...     bending_weights=[0.02, 0.01, 0.005],
    ... )
    """
    n_levels = len(grid_spacings)
    
    # Set defaults
    if iterations_per_level is None:
        iterations_per_level = [300, 200, 100][:n_levels]
        if len(iterations_per_level) < n_levels:
            iterations_per_level.extend([100] * (n_levels - len(iterations_per_level)))
    
    # Normalize smooth_weights to list
    if smooth_weights is None:
        smooth_weights = [0.01] * n_levels
    elif isinstance(smooth_weights, (int, float)):
        smooth_weights = [float(smooth_weights)] * n_levels
    
    # Normalize bending_weights to list
    if bending_weights is None:
        bending_weights = [0.01] * n_levels
    elif isinstance(bending_weights, (int, float)):
        bending_weights = [float(bending_weights)] * n_levels
    
    # Normalize grid_spacings to list of tuples
    normalized_spacings = [_normalize_grid_spacing(gs, 3) for gs in grid_spacings]
    
    # Validate lengths
    if len(iterations_per_level) != n_levels:
        raise ValueError(f"iterations_per_level length ({len(iterations_per_level)}) must match grid_spacings ({n_levels})")
    if len(smooth_weights) != n_levels:
        raise ValueError(f"smooth_weights length ({len(smooth_weights)}) must match grid_spacings ({n_levels})")
    if len(bending_weights) != n_levels:
        raise ValueError(f"bending_weights length ({len(bending_weights)}) must match grid_spacings ({n_levels})")
    
    displacements = []
    level_infos = []
    total_iterations = 0
    max_displacements = []
    
    if verbose:
        print(f"Multiscale 3D registration with {n_levels} levels")
        print(f"Grid spacings (Z, Y, X): {normalized_spacings}")
    
    # Start with moving volume
    current_moving = moving.astype(np.float32)
    
    for level_idx in range(n_levels):
        spacing = normalized_spacings[level_idx]
        n_iter = iterations_per_level[level_idx]
        smooth_w = smooth_weights[level_idx]
        bending_w = bending_weights[level_idx]
        
        if verbose:
            print(f"\n--- Level {level_idx + 1}/{n_levels} (spacing={spacing}, "
                  f"smooth={smooth_w:.4f}, bending={bending_w:.4f}) ---")
        
        # Register at this level using per-axis spacing
        disp, _, _, _, info = register_ffd_3d(
            current_moving, fixed,
            grid_spacing=spacing,
            smooth_weight=smooth_w,
            bending_weight=bending_w,
            n_iterations=n_iter,
            lr=lr,
            mask=mask,
            loss_fn=loss_fn,
            device=device,
            verbose=verbose,
        )
        
        displacements.append(disp)
        level_infos.append(info)
        total_iterations += info['iterations']
        max_displacements.append(info['max_displacement'])
        
        # Apply this displacement to get the new moving volume for next level
        current_moving = apply_ffd_3d(current_moving, disp, device=device)
    
    # Final info
    result_info = {
        'grid_spacings': normalized_spacings,
        'iterations_per_level': iterations_per_level,
        'smooth_weights': smooth_weights,
        'bending_weights': bending_weights,
        'level_info': level_infos,
        'total_iterations': total_iterations,
        'max_displacements': max_displacements,
    }
    
    if verbose:
        print(f"\nMultiscale complete. Total iterations: {total_iterations}")
        print(f"Max displacement per level: {[f'{d:.1f}' for d in max_displacements]}")
    
    return displacements, result_info

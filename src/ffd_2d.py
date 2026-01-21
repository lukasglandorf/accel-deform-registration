# -*- coding: utf-8 -*-
"""
2D Free-Form Deformation (FFD) registration.

This module provides B-spline FFD registration for 2D images using PyTorch.
Optimized for MIP (Maximum Intensity Projection) images from OCT volumes.

Functions
---------
1. register_ffd_2d: Main 2D FFD registration function
2. apply_ffd_2d: Apply 2D displacement field to an image
3. create_2d_grid_image: Visualize the control point grid
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, Union
from numpy.typing import NDArray

from .ffd_common import get_default_device, normalize_image, compute_validity_mask_2d
from .losses import BaseLoss, CorrelationLoss


def register_ffd_2d(
    moving: NDArray[np.floating],
    fixed: NDArray[np.floating],
    grid_spacing: Union[int, Tuple[int, int]] = 100,
    smooth_weight: float = 0.005,
    bending_weight: float = 0.01,
    n_iterations: int = 2000,
    lr: float = 0.5,
    padding_mode: str = 'border',
    use_boundary_layer: bool = True,
    loss_fn: Optional[BaseLoss] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], 
           NDArray[np.bool_], Dict[str, Any]]:
    """
    Perform 2D FFD registration using B-spline control points.
    
    Optimizes a displacement field to align the moving image to the fixed image
    using correlation as the similarity metric.
    
    Parameters
    ----------
    moving : ndarray
        Moving (source) image to be warped, shape (Y, X).
    fixed : ndarray
        Fixed (target) image, shape (Y, X).
    grid_spacing : int or tuple of int
        Spacing between control points in pixels. Can be a single int (same
        for both axes) or tuple (spacing_y, spacing_x). Smaller values allow
        finer deformations but increase computation. Default 100.
    smooth_weight : float
        Weight for displacement magnitude regularization. Higher values
        penalize large displacements. Default 0.005.
    bending_weight : float
        Weight for bending energy (second derivative) regularization.
        Higher values enforce smoother deformations. Default 0.01.
    n_iterations : int
        Number of optimization iterations. Default 2000.
    lr : float
        Learning rate for Adam optimizer. Default 0.5.
    padding_mode : str
        Padding mode for grid_sample: 'border', 'zeros', or 'reflection'.
        Default 'border' (recommended for avoiding black rim artifacts).
    use_boundary_layer : bool
        If True (default), add an extra layer of control points outside the
        image boundaries to reduce edge effects. If False, control points
        are placed only within the image extent.
    loss_fn : BaseLoss, optional
        Loss function for similarity measurement. If None, uses CorrelationLoss.
        Available: CorrelationLoss, MAELoss, MutualInformationLoss.
    device : torch.device, optional
        PyTorch device. If None, auto-detect (CUDA > MPS > CPU).
    verbose : bool
        Print progress information. Default True.
    
    Returns
    -------
    displacement_field : ndarray
        Dense displacement field of shape (Y, X, 2) in pixel units.
        displacement_field[y, x] = [dx, dy] means pixel (x, y) in the
        warped image comes from (x - dx, y - dy) in the moving image.
    control_point_displacements : ndarray
        Control point displacements of shape (n_ctrl_y, n_ctrl_x, 2).
    control_point_positions : ndarray
        Original control point positions of shape (n_ctrl_y, n_ctrl_x, 2).
    validity_mask : ndarray
        Boolean mask of shape (Y, X) indicating which pixels have valid
        source locations (not outside image boundaries).
    info : dict
        Registration metadata including:
        - 'iterations': Number of iterations completed
        - 'initial_correlation': Correlation before registration
        - 'final_correlation': Correlation after registration
        - 'loss_history': List of loss values per iteration
        - 'max_displacement': Maximum displacement magnitude
        - 'grid_dims': Control grid dimensions (n_ctrl_x, n_ctrl_y)
        - 'grid_spacing': Actual grid spacing used
        - 'spacing': Tuple of (spacing_x, spacing_y) in pixels
    
    Notes
    -----
    When use_boundary_layer=True (default), the control grid includes an
    extra boundary layer (+1 on each side) to mitigate edge effects. The
    returned control_point_displacements contains only the interior points.
    When use_boundary_layer=False, control points are placed only within
    the image extent.
    
    Examples
    --------
    >>> disp, ctrl_disp, ctrl_pos, mask, info = register_ffd_2d(
    ...     moving_mip, fixed_mip, grid_spacing=100, n_iterations=2000
    ... )
    >>> print(f"Correlation: {info['initial_correlation']:.4f} -> "
    ...       f"{info['final_correlation']:.4f}")
    """
    device = get_default_device(device)
    
    # Normalize images
    moving_np = normalize_image(moving)
    fixed_np = normalize_image(fixed)
    
    # Handle different sized images (corner-aligned cropping)
    # Crop to minimum common size from top-left corner
    min_y = min(moving_np.shape[0], fixed_np.shape[0])
    min_x = min(moving_np.shape[1], fixed_np.shape[1])
    
    if moving_np.shape != fixed_np.shape:
        if verbose:
            print(f"    Note: Image sizes differ - moving: {moving_np.shape}, fixed: {fixed_np.shape}")
            print(f"    Cropping to common size: ({min_y}, {min_x})")
        moving_np = moving_np[:min_y, :min_x]
        fixed_np = fixed_np[:min_y, :min_x]
    
    Y, X = moving_np.shape
    
    # Parse grid_spacing (can be int or (spacing_y, spacing_x) tuple)
    if isinstance(grid_spacing, (int, np.integer)):
        spacing_target_y = int(grid_spacing)
        spacing_target_x = int(grid_spacing)
    else:
        if len(grid_spacing) != 2:
            raise ValueError(f"grid_spacing tuple must have 2 elements, got {len(grid_spacing)}")
        spacing_target_y, spacing_target_x = int(grid_spacing[0]), int(grid_spacing[1])
    
    # Compute control grid dimensions from target spacing
    n_ctrl_x = max(2, X // spacing_target_x + 1)
    n_ctrl_y = max(2, Y // spacing_target_y + 1)
    
    # Extended grid with boundary layer (optional)
    if use_boundary_layer:
        n_full_x = n_ctrl_x + 2
        n_full_y = n_ctrl_y + 2
    else:
        n_full_x = n_ctrl_x
        n_full_y = n_ctrl_y
    
    # Actual spacing between control points
    spacing_x = (X - 1) / (n_ctrl_x - 1) if n_ctrl_x > 1 else float(X)
    spacing_y = (Y - 1) / (n_ctrl_y - 1) if n_ctrl_y > 1 else float(Y)
    
    # Initialize loss function
    if loss_fn is None:
        loss_fn = CorrelationLoss()
    
    if verbose:
        print(f"    Image shape: {X}x{Y}")
        if use_boundary_layer:
            print(f"    Control grid: {n_ctrl_x}x{n_ctrl_y} + boundary = {n_full_x}x{n_full_y}")
        else:
            print(f"    Control grid: {n_ctrl_x}x{n_ctrl_y} (no boundary layer)")
        print(f"    Target spacing: {spacing_target_x}x{spacing_target_y} -> actual: {spacing_x:.1f}x{spacing_y:.1f} pixels")
        print(f"    Loss function: {loss_fn.name}")
    
    # Store original control point positions (interior only)
    ctrl_positions = np.zeros((n_ctrl_y, n_ctrl_x, 2), dtype=np.float32)
    for iy in range(n_ctrl_y):
        for ix in range(n_ctrl_x):
            ctrl_positions[iy, ix, 0] = ix * spacing_x  # x
            ctrl_positions[iy, ix, 1] = iy * spacing_y  # y
    
    # Convert to tensors
    moving_t = torch.from_numpy(moving_np).unsqueeze(0).unsqueeze(0).to(device)
    fixed_t = torch.from_numpy(fixed_np).unsqueeze(0).unsqueeze(0).to(device)
    
    # Initialize control point displacements (extended grid)
    ctrl_disps = torch.zeros(1, 2, n_full_y, n_full_x, device=device, requires_grad=True)
    max_disp = min(spacing_x, spacing_y) / 2.0  # Use actual spacing for normalization
    
    optimizer = torch.optim.Adam([ctrl_disps], lr=lr)
    losses = []
    
    # Create base sampling grid (normalized to [-1, 1])
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, Y, device=device),
        torch.linspace(-1, 1, X, device=device),
        indexing='ij'
    )
    base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)
    
    # Compute sampling grid (maps image FOV to control grid)
    def compute_sample_grid():
        if use_boundary_layer:
            # Map image FOV to interior of extended control grid
            x_min_ext, x_max_ext = -spacing_x, X - 1 + spacing_x
            y_min_ext, y_max_ext = -spacing_y, Y - 1 + spacing_y
        else:
            # Map image FOV to full control grid extent
            x_min_ext, x_max_ext = 0, X - 1
            y_min_ext, y_max_ext = 0, Y - 1
        
        def to_norm(val, vmin, vmax):
            return 2.0 * (val - vmin) / (vmax - vmin) - 1.0
        
        x_norm_min = to_norm(0, x_min_ext, x_max_ext)
        x_norm_max = to_norm(X - 1, x_min_ext, x_max_ext)
        y_norm_min = to_norm(0, y_min_ext, y_max_ext)
        y_norm_max = to_norm(Y - 1, y_min_ext, y_max_ext)
        
        yy_int, xx_int = torch.meshgrid(
            torch.linspace(y_norm_min, y_norm_max, Y, device=device),
            torch.linspace(x_norm_min, x_norm_max, X, device=device),
            indexing='ij'
        )
        return torch.stack([xx_int, yy_int], dim=-1).unsqueeze(0)
    
    sample_grid = compute_sample_grid()
    initial_loss = None
    
    # Optimization loop
    for i in range(n_iterations):
        optimizer.zero_grad()
        
        # Upsample control point displacements to full resolution
        disp_full = F.grid_sample(
            ctrl_disps, sample_grid, mode='bilinear',
            padding_mode=padding_mode, align_corners=True
        )
        
        # Normalize displacements to [-1, 1] grid coordinates
        disp_normalized = disp_full.clone()
        disp_normalized[:, 0] = disp_full[:, 0] / (X / 2)
        disp_normalized[:, 1] = disp_full[:, 1] / (Y / 2)
        
        # Apply displacement to sampling grid
        disp_grid = disp_normalized.permute(0, 2, 3, 1)
        deformed_grid = base_grid + disp_grid
        
        # Warp moving image
        warped = F.grid_sample(
            moving_t, deformed_grid, mode='bilinear',
            padding_mode=padding_mode, align_corners=True
        )
        
        # Similarity loss (using pluggable loss function)
        loss_similarity = loss_fn(warped, fixed_t)
        
        # Smoothness penalty (displacement magnitude)
        disp_magnitude = torch.sqrt((ctrl_disps ** 2).sum(dim=1, keepdim=True) + 1e-8)
        normalized_disp = disp_magnitude / max_disp
        loss_smooth = smooth_weight * (normalized_disp ** 2).mean()
        
        # Bending energy (second derivatives)
        loss_bending = torch.tensor(0.0, device=device)
        if ctrl_disps.shape[3] > 2:
            d2_dx2 = ctrl_disps[:, :, :, 2:] - 2*ctrl_disps[:, :, :, 1:-1] + ctrl_disps[:, :, :, :-2]
            loss_bending = loss_bending + (d2_dx2 ** 2).mean()
        if ctrl_disps.shape[2] > 2:
            d2_dy2 = ctrl_disps[:, :, 2:, :] - 2*ctrl_disps[:, :, 1:-1, :] + ctrl_disps[:, :, :-2, :]
            loss_bending = loss_bending + (d2_dy2 ** 2).mean()
        loss_bend = bending_weight * loss_bending
        
        # Total loss
        loss = loss_similarity + loss_smooth + loss_bend
        
        if initial_loss is None:
            initial_loss = loss.item()
        
        if torch.isnan(loss):
            if verbose:
                print(f"    Warning: NaN detected at iter {i+1}, stopping")
            break
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_([ctrl_disps], max_norm=10.0)
        optimizer.step()
        losses.append(loss.item())
        
        if verbose and (i + 1) % 50 == 0:
            max_d = disp_magnitude.max().item()
            print(f"      Iter {i+1}: loss={loss.item():.4f} "
                  f"(corr={-loss_similarity.item():.4f}, smooth={loss_smooth.item():.4f}, "
                  f"bend={loss_bend.item():.4f}, max_d={max_d:.1f}px)")
    
    # Extract final displacement field
    with torch.no_grad():
        disp_full = F.grid_sample(
            ctrl_disps, sample_grid, mode='bilinear',
            padding_mode=padding_mode, align_corners=True
        )
        disp_voxels = disp_full.squeeze().permute(1, 2, 0).cpu().numpy()
    
    # Extract control point displacements (interior only if boundary layer used)
    if use_boundary_layer:
        ctrl_disps_out = ctrl_disps[:, :, 1:-1, 1:-1].squeeze().permute(1, 2, 0).detach().cpu().numpy()
    else:
        ctrl_disps_out = ctrl_disps.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    
    # Compute validity mask
    validity_mask = compute_validity_mask_2d(disp_voxels, (Y, X), margin=1.0)
    
    info = {
        "iterations": len(losses),
        "initial_loss": initial_loss,
        "final_loss": losses[-1] if losses else 0.0,
        "initial_correlation": -initial_loss if initial_loss else 0.0,
        "final_correlation": -losses[-1] if losses else 0.0,
        "loss_history": losses,
        "max_displacement": float(np.linalg.norm(disp_voxels.reshape(-1, 2), axis=1).max()),
        "grid_dims": (n_ctrl_x, n_ctrl_y),
        "grid_dims_full": (n_full_x, n_full_y),
        "grid_spacing": (spacing_target_y, spacing_target_x),  # Requested spacing
        "spacing": (spacing_x, spacing_y),  # Actual computed spacing
        "use_boundary_layer": use_boundary_layer,
    }
    
    if verbose:
        print(f"    Correlation: {info['initial_correlation']:.4f} → {info['final_correlation']:.4f}")
        print(f"    Max displacement: {info['max_displacement']:.1f} pixels")
    
    return disp_voxels, ctrl_disps_out, ctrl_positions, validity_mask, info


def apply_ffd_2d(
    image: NDArray[np.floating],
    displacement_field: NDArray[np.floating],
    padding_mode: str = 'border',
    device: Optional[torch.device] = None,
) -> NDArray:
    """
    Apply a 2D displacement field to warp an image.
    
    Parameters
    ----------
    image : ndarray
        Input image of shape (Y, X).
    displacement_field : ndarray
        Displacement field of shape (Y, X, 2) in pixel units.
    padding_mode : str
        Padding mode: 'border', 'zeros', or 'reflection'. Default 'border'.
    device : torch.device, optional
        PyTorch device. If None, auto-detect.
    
    Returns
    -------
    warped : ndarray
        Warped image of shape (Y, X), same dtype as input.
    """
    device = get_default_device(device)
    
    Y, X = image.shape
    original_dtype = image.dtype
    
    # Normalize image
    img_min, img_max = image.min(), image.max()
    img_norm = (image.astype(np.float32) - img_min) / (img_max - img_min + 1e-8)
    
    # Convert to tensors
    img_t = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)
    
    # Create sampling grid
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, Y, device=device),
        torch.linspace(-1, 1, X, device=device),
        indexing='ij'
    )
    base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)
    
    # Normalize displacement to grid coordinates
    disp_t = torch.from_numpy(displacement_field.astype(np.float32)).to(device)
    disp_norm = torch.zeros_like(disp_t)
    disp_norm[..., 0] = disp_t[..., 0] / (X / 2)
    disp_norm[..., 1] = disp_t[..., 1] / (Y / 2)
    
    # Apply displacement
    deformed_grid = base_grid + disp_norm.unsqueeze(0)
    warped_t = F.grid_sample(
        img_t, deformed_grid, mode='bilinear',
        padding_mode=padding_mode, align_corners=True
    )
    
    # Convert back
    warped = warped_t.squeeze().cpu().numpy()
    warped = warped * (img_max - img_min) + img_min
    
    return warped.astype(original_dtype)


def create_2d_grid_image(
    control_point_displacements: NDArray[np.floating],
    control_point_positions: NDArray[np.floating],
    image_shape: tuple[int, int],
    point_value: int = 255,
    line_value: int = 128,
) -> NDArray[np.uint8]:
    """
    Create a visualization of the 2D FFD control grid.
    
    Parameters
    ----------
    control_point_displacements : ndarray
        Control point displacements of shape (n_ctrl_y, n_ctrl_x, 2).
    control_point_positions : ndarray
        Original control point positions of shape (n_ctrl_y, n_ctrl_x, 2).
    image_shape : tuple
        Output image shape (Y, X).
    point_value : int
        Pixel value for control points. Default 255.
    line_value : int
        Pixel value for grid lines. Default 128.
    
    Returns
    -------
    grid_image : ndarray
        Visualization image of shape (Y, X) with dtype uint8.
    """
    Y, X = image_shape
    n_ctrl_y, n_ctrl_x = control_point_displacements.shape[:2]
    
    image = np.zeros((Y, X), dtype=np.uint8)
    
    # Compute deformed positions
    deformed_positions = control_point_positions + control_point_displacements
    
    # Draw lines helper
    def draw_line(p1, p2, img, val):
        p1, p2 = np.array(p1), np.array(p2)
        diff = p2 - p1
        max_steps = int(np.ceil(np.linalg.norm(diff))) + 1
        for t in np.linspace(0, 1, max_steps):
            pt = p1 + t * diff
            x, y = int(round(pt[0])), int(round(pt[1]))
            if 0 <= x < X and 0 <= y < Y:
                if img[y, x] != point_value:  # Don't overwrite control points
                    img[y, x] = val
    
    # Draw horizontal lines
    for iy in range(n_ctrl_y):
        for ix in range(n_ctrl_x - 1):
            draw_line(
                deformed_positions[iy, ix],
                deformed_positions[iy, ix + 1],
                image, line_value
            )
    
    # Draw vertical lines
    for iy in range(n_ctrl_y - 1):
        for ix in range(n_ctrl_x):
            draw_line(
                deformed_positions[iy, ix],
                deformed_positions[iy + 1, ix],
                image, line_value
            )
    
    # Draw control points (on top)
    for iy in range(n_ctrl_y):
        for ix in range(n_ctrl_x):
            x, y = deformed_positions[iy, ix]
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < X and 0 <= yi < Y:
                image[yi, xi] = point_value
    
    return image

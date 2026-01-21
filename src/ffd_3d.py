# -*- coding: utf-8 -*-
"""
3D Free-Form Deformation (FFD) registration.

This module provides B-spline FFD registration for 3D volumes using PyTorch.
Optimized for OCT volume registration with Z, Y, X axis ordering.

Functions
---------
1. register_ffd_3d: Main 3D FFD registration function
2. apply_ffd_3d: Apply 3D displacement field to a volume
3. create_3d_grid_image: Visualize the control point grid (MIP projection)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, Union
from numpy.typing import NDArray

from .ffd_common import get_default_device, normalize_image, compute_validity_mask_3d
from .losses import BaseLoss, CorrelationLoss


def register_ffd_3d(
    moving: NDArray[np.floating],
    fixed: NDArray[np.floating],
    grid_spacing: Union[int, Tuple[int, int, int]] = 40,
    smooth_weight: float = 0.01,
    bending_weight: float = 0.01,
    n_iterations: int = 400,
    lr: float = 0.5,
    padding_mode: str = 'border',
    use_boundary_layer: bool = True,
    mask: Optional[NDArray[np.floating]] = None,
    loss_fn: Optional[BaseLoss] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], 
           NDArray[np.bool_], Dict[str, Any]]:
    """
    Perform 3D FFD registration using B-spline control points.
    
    Optimizes a displacement field to align the moving volume to the fixed
    volume using correlation as the similarity metric.
    
    Parameters
    ----------
    moving : ndarray
        Moving (source) volume to be warped, shape (Z, Y, X).
    fixed : ndarray
        Fixed (target) volume, shape (Z, Y, X).
    grid_spacing : int or tuple of int
        Spacing between control points in voxels. Can be a single int (same
        for all axes) or tuple (spacing_z, spacing_y, spacing_x). Smaller 
        values allow finer deformations but increase computation. Default 40.
    smooth_weight : float
        Weight for displacement magnitude regularization. Default 0.01.
    bending_weight : float
        Weight for bending energy regularization. Default 0.01.
    n_iterations : int
        Number of optimization iterations. Default 400.
    lr : float
        Learning rate for Adam optimizer. Default 0.5.
    padding_mode : str
        Padding mode for grid_sample: 'border', 'zeros', or 'reflection'.
        Default 'border' (recommended for OCT volumes).
    use_boundary_layer : bool
        If True (default), add an extra layer of control points outside the
        volume boundaries to reduce edge effects. If False, control points
        are placed only within the volume extent.
    mask : ndarray, optional
        Optional mask of shape (Z, Y, X) for weighted loss computation.
        Values should be in [0, 1] where 1 indicates full weight.
        If None, all voxels contribute equally to the loss.
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
        Dense displacement field of shape (Z, Y, X, 3) in voxel units.
        displacement_field[z, y, x] = [dx, dy, dz] where the ordering
        matches PyTorch grid_sample convention (x, y, z).
    control_point_displacements : ndarray
        Control point displacements of shape (n_ctrl_z, n_ctrl_y, n_ctrl_x, 3).
    control_point_positions : ndarray
        Original control point positions of shape (n_ctrl_z, n_ctrl_y, n_ctrl_x, 3).
    validity_mask : ndarray
        Boolean mask of shape (Z, Y, X) indicating which voxels have valid
        source locations.
    info : dict
        Registration metadata including:
        - 'iterations': Number of iterations completed
        - 'initial_correlation': Correlation before registration
        - 'final_correlation': Correlation after registration
        - 'loss_history': List of loss values per iteration
        - 'max_displacement': Maximum displacement magnitude
        - 'grid_dims': Control grid dimensions (n_ctrl_x, n_ctrl_y, n_ctrl_z)
        - 'spacing': Tuple of (spacing_x, spacing_y, spacing_z) in voxels
    
    Notes
    -----
    When use_boundary_layer=True (default), the control grid includes an
    extra boundary layer (+1 on each side) to mitigate edge effects.
    When use_boundary_layer=False, control points are placed only within
    the volume extent.
    
    Examples
    --------
    >>> disp, ctrl_disp, ctrl_pos, mask, info = register_ffd_3d(
    ...     moving_vol, fixed_vol, grid_spacing=40, n_iterations=400
    ... )
    >>> print(f"Correlation: {info['initial_correlation']:.4f} -> "
    ...       f"{info['final_correlation']:.4f}")
    """
    device = get_default_device(device)
    
    # Normalize images
    moving_np = normalize_image(moving)
    fixed_np = normalize_image(fixed)
    
    # Handle different sized volumes (corner-aligned cropping)
    # Crop to minimum common size from origin corner
    min_z = min(moving_np.shape[0], fixed_np.shape[0])
    min_y = min(moving_np.shape[1], fixed_np.shape[1])
    min_x = min(moving_np.shape[2], fixed_np.shape[2])
    
    if moving_np.shape != fixed_np.shape:
        if verbose:
            print(f"    Note: Volume sizes differ - moving: {moving_np.shape}, fixed: {fixed_np.shape}")
            print(f"    Cropping to common size: ({min_z}, {min_y}, {min_x})")
        moving_np = moving_np[:min_z, :min_y, :min_x]
        fixed_np = fixed_np[:min_z, :min_y, :min_x]
        
        # Also crop mask if provided
        if mask is not None:
            mask = mask[:min_z, :min_y, :min_x]
    
    Z, Y, X = moving_np.shape
    
    # Parse grid_spacing (can be int or (spacing_z, spacing_y, spacing_x) tuple)
    if isinstance(grid_spacing, (int, np.integer)):
        spacing_target_z = int(grid_spacing)
        spacing_target_y = int(grid_spacing)
        spacing_target_x = int(grid_spacing)
    else:
        if len(grid_spacing) != 3:
            raise ValueError(f"grid_spacing tuple must have 3 elements, got {len(grid_spacing)}")
        spacing_target_z, spacing_target_y, spacing_target_x = (
            int(grid_spacing[0]), int(grid_spacing[1]), int(grid_spacing[2])
        )
    
    # Compute control grid dimensions from target spacing
    n_ctrl_x = max(2, X // spacing_target_x + 1)
    n_ctrl_y = max(2, Y // spacing_target_y + 1)
    n_ctrl_z = max(2, Z // spacing_target_z + 1)
    
    # Extended grid with boundary layer (optional)
    if use_boundary_layer:
        n_full_x = n_ctrl_x + 2
        n_full_y = n_ctrl_y + 2
        n_full_z = n_ctrl_z + 2
    else:
        n_full_x = n_ctrl_x
        n_full_y = n_ctrl_y
        n_full_z = n_ctrl_z
    
    # Actual spacing between control points
    spacing_x = (X - 1) / (n_ctrl_x - 1) if n_ctrl_x > 1 else float(X)
    spacing_y = (Y - 1) / (n_ctrl_y - 1) if n_ctrl_y > 1 else float(Y)
    spacing_z = (Z - 1) / (n_ctrl_z - 1) if n_ctrl_z > 1 else float(Z)
    
    # Initialize loss function
    if loss_fn is None:
        loss_fn = CorrelationLoss()
    
    if verbose:
        print(f"    Volume shape: {X}x{Y}x{Z}")
        if use_boundary_layer:
            print(f"    Control grid: {n_ctrl_x}x{n_ctrl_y}x{n_ctrl_z} + boundary = "
                  f"{n_full_x}x{n_full_y}x{n_full_z}")
        else:
            print(f"    Control grid: {n_ctrl_x}x{n_ctrl_y}x{n_ctrl_z} (no boundary layer)")
        print(f"    Target spacing: {spacing_target_x}x{spacing_target_y}x{spacing_target_z} -> "
              f"actual: {spacing_x:.1f}x{spacing_y:.1f}x{spacing_z:.1f} voxels")
        print(f"    Loss function: {loss_fn.name}")
    
    # Store original control point positions (interior only)
    ctrl_positions = np.zeros((n_ctrl_z, n_ctrl_y, n_ctrl_x, 3), dtype=np.float32)
    for iz in range(n_ctrl_z):
        for iy in range(n_ctrl_y):
            for ix in range(n_ctrl_x):
                ctrl_positions[iz, iy, ix, 0] = ix * spacing_x  # x
                ctrl_positions[iz, iy, ix, 1] = iy * spacing_y  # y
                ctrl_positions[iz, iy, ix, 2] = iz * spacing_z  # z
    
    # Convert to tensors (add batch and channel dimensions)
    # PyTorch 3D grid_sample expects: (N, C, D, H, W) = (N, C, Z, Y, X)
    moving_t = torch.from_numpy(moving_np).unsqueeze(0).unsqueeze(0).to(device)
    fixed_t = torch.from_numpy(fixed_np).unsqueeze(0).unsqueeze(0).to(device)
    
    # Convert mask to tensor if provided
    mask_t = None
    if mask is not None:
        mask_np = mask.astype(np.float32)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
        if verbose:
            mask_coverage = mask_np.sum() / mask_np.size * 100
            print(f"    Using mask: {mask_coverage:.1f}% coverage")
    
    # Initialize control point displacements (extended grid)
    # Shape: (N, 3, D, H, W) = (1, 3, n_full_z, n_full_y, n_full_x)
    ctrl_disps = torch.zeros(1, 3, n_full_z, n_full_y, n_full_x, device=device, requires_grad=True)
    max_disp = min(spacing_x, spacing_y, spacing_z) / 2.0  # Use actual spacing for normalization
    
    optimizer = torch.optim.Adam([ctrl_disps], lr=lr)
    losses = []
    
    # Create base sampling grid (normalized to [-1, 1])
    # PyTorch 3D grid_sample grid shape: (N, D, H, W, 3) with order (x, y, z)
    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, Z, device=device),
        torch.linspace(-1, 1, Y, device=device),
        torch.linspace(-1, 1, X, device=device),
        indexing='ij'
    )
    base_grid = torch.stack([xx, yy, zz], dim=-1).unsqueeze(0)  # (1, Z, Y, X, 3)
    
    # Compute sampling grid (maps volume FOV to control grid)
    def compute_sample_grid():
        if use_boundary_layer:
            # Map volume FOV to interior of extended control grid
            x_min_ext, x_max_ext = -spacing_x, X - 1 + spacing_x
            y_min_ext, y_max_ext = -spacing_y, Y - 1 + spacing_y
            z_min_ext, z_max_ext = -spacing_z, Z - 1 + spacing_z
        else:
            # Map volume FOV to full control grid extent
            x_min_ext, x_max_ext = 0, X - 1
            y_min_ext, y_max_ext = 0, Y - 1
            z_min_ext, z_max_ext = 0, Z - 1
        
        def to_norm(val, vmin, vmax):
            return 2.0 * (val - vmin) / (vmax - vmin) - 1.0
        
        x_norm_min = to_norm(0, x_min_ext, x_max_ext)
        x_norm_max = to_norm(X - 1, x_min_ext, x_max_ext)
        y_norm_min = to_norm(0, y_min_ext, y_max_ext)
        y_norm_max = to_norm(Y - 1, y_min_ext, y_max_ext)
        z_norm_min = to_norm(0, z_min_ext, z_max_ext)
        z_norm_max = to_norm(Z - 1, z_min_ext, z_max_ext)
        
        zz_int, yy_int, xx_int = torch.meshgrid(
            torch.linspace(z_norm_min, z_norm_max, Z, device=device),
            torch.linspace(y_norm_min, y_norm_max, Y, device=device),
            torch.linspace(x_norm_min, x_norm_max, X, device=device),
            indexing='ij'
        )
        return torch.stack([xx_int, yy_int, zz_int], dim=-1).unsqueeze(0)
    
    sample_grid = compute_sample_grid()  # (1, Z, Y, X, 3)
    initial_loss = None
    
    # Optimization loop
    for i in range(n_iterations):
        optimizer.zero_grad()
        
        # Upsample control point displacements to full resolution
        # grid_sample expects: (N, D, H, W, 3) grid
        disp_full = F.grid_sample(
            ctrl_disps, sample_grid, mode='bilinear',
            padding_mode=padding_mode, align_corners=True
        )  # (1, 3, Z, Y, X)
        
        # Normalize displacements to [-1, 1] grid coordinates
        disp_normalized = disp_full.clone()
        disp_normalized[:, 0] = disp_full[:, 0] / (X / 2)  # dx
        disp_normalized[:, 1] = disp_full[:, 1] / (Y / 2)  # dy
        disp_normalized[:, 2] = disp_full[:, 2] / (Z / 2)  # dz
        
        # Apply displacement to sampling grid
        disp_grid = disp_normalized.permute(0, 2, 3, 4, 1)  # (1, Z, Y, X, 3)
        deformed_grid = base_grid + disp_grid
        
        # Warp moving volume
        warped = F.grid_sample(
            moving_t, deformed_grid, mode='bilinear',
            padding_mode=padding_mode, align_corners=True
        )
        
        # Similarity loss (using pluggable loss function)
        loss_similarity = loss_fn(warped, fixed_t, mask_t)
        
        # Smoothness penalty (displacement magnitude)
        disp_magnitude = torch.sqrt((ctrl_disps ** 2).sum(dim=1, keepdim=True) + 1e-8)
        normalized_disp = disp_magnitude / max_disp
        loss_smooth = smooth_weight * (normalized_disp ** 2).mean()
        
        # Bending energy (second derivatives in all directions)
        loss_bending = torch.tensor(0.0, device=device)
        
        # d^2/dx^2
        if ctrl_disps.shape[4] > 2:
            d2_dx2 = ctrl_disps[:, :, :, :, 2:] - 2*ctrl_disps[:, :, :, :, 1:-1] + ctrl_disps[:, :, :, :, :-2]
            loss_bending = loss_bending + (d2_dx2 ** 2).mean()
        # d^2/dy^2
        if ctrl_disps.shape[3] > 2:
            d2_dy2 = ctrl_disps[:, :, :, 2:, :] - 2*ctrl_disps[:, :, :, 1:-1, :] + ctrl_disps[:, :, :, :-2, :]
            loss_bending = loss_bending + (d2_dy2 ** 2).mean()
        # d^2/dz^2
        if ctrl_disps.shape[2] > 2:
            d2_dz2 = ctrl_disps[:, :, 2:, :, :] - 2*ctrl_disps[:, :, 1:-1, :, :] + ctrl_disps[:, :, :-2, :, :]
            loss_bending = loss_bending + (d2_dz2 ** 2).mean()
        
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
                  f"bend={loss_bend.item():.4f}, max_d={max_d:.1f}vox)")
    
    # Extract final displacement field
    with torch.no_grad():
        disp_full = F.grid_sample(
            ctrl_disps, sample_grid, mode='bilinear',
            padding_mode=padding_mode, align_corners=True
        )
        # Shape: (1, 3, Z, Y, X) -> (Z, Y, X, 3)
        disp_voxels = disp_full.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    
    # Extract control point displacements (interior only if boundary layer used)
    # ctrl_disps: (1, 3, n_full_z, n_full_y, n_full_x)
    if use_boundary_layer:
        ctrl_disps_out = ctrl_disps[:, :, 1:-1, 1:-1, 1:-1].squeeze(0).permute(1, 2, 3, 0).detach().cpu().numpy()
    else:
        ctrl_disps_out = ctrl_disps.squeeze(0).permute(1, 2, 3, 0).detach().cpu().numpy()
    
    # Compute validity mask
    validity_mask = compute_validity_mask_3d(disp_voxels, (Z, Y, X), margin=1.0)
    
    info = {
        "iterations": len(losses),
        "initial_loss": initial_loss,
        "final_loss": losses[-1] if losses else 0.0,
        "initial_correlation": -initial_loss if initial_loss else 0.0,
        "final_correlation": -losses[-1] if losses else 0.0,
        "loss_history": losses,
        "max_displacement": float(np.linalg.norm(disp_voxels.reshape(-1, 3), axis=1).max()),
        "grid_dims": (n_ctrl_x, n_ctrl_y, n_ctrl_z),
        "grid_dims_full": (n_full_x, n_full_y, n_full_z),
        "grid_spacing": (spacing_target_z, spacing_target_y, spacing_target_x),  # Requested spacing
        "spacing": (spacing_x, spacing_y, spacing_z),  # Actual computed spacing
        "use_boundary_layer": use_boundary_layer,
    }
    
    if verbose:
        print(f"    Correlation: {info['initial_correlation']:.4f} → {info['final_correlation']:.4f}")
        print(f"    Max displacement: {info['max_displacement']:.1f} voxels")
    
    return disp_voxels, ctrl_disps_out, ctrl_positions, validity_mask, info


def apply_ffd_3d(
    volume: NDArray[np.floating],
    displacement_field: NDArray[np.floating],
    padding_mode: str = 'border',
    device: Optional[torch.device] = None,
) -> NDArray:
    """
    Apply a 3D displacement field to warp a volume.
    
    Parameters
    ----------
    volume : ndarray
        Input volume of shape (Z, Y, X).
    displacement_field : ndarray
        Displacement field of shape (Z, Y, X, 3) in voxel units.
    padding_mode : str
        Padding mode: 'border', 'zeros', or 'reflection'. Default 'border'.
    device : torch.device, optional
        PyTorch device. If None, auto-detect.
    
    Returns
    -------
    warped : ndarray
        Warped volume of shape (Z, Y, X), same dtype as input.
    """
    device = get_default_device(device)
    
    Z, Y, X = volume.shape
    original_dtype = volume.dtype
    
    # Normalize volume
    vol_min, vol_max = volume.min(), volume.max()
    vol_norm = (volume.astype(np.float32) - vol_min) / (vol_max - vol_min + 1e-8)
    
    # Convert to tensors
    vol_t = torch.from_numpy(vol_norm).unsqueeze(0).unsqueeze(0).to(device)
    
    # Create sampling grid
    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, Z, device=device),
        torch.linspace(-1, 1, Y, device=device),
        torch.linspace(-1, 1, X, device=device),
        indexing='ij'
    )
    base_grid = torch.stack([xx, yy, zz], dim=-1).unsqueeze(0)
    
    # Normalize displacement to grid coordinates
    disp_t = torch.from_numpy(displacement_field.astype(np.float32)).to(device)
    disp_norm = torch.zeros_like(disp_t)
    disp_norm[..., 0] = disp_t[..., 0] / (X / 2)
    disp_norm[..., 1] = disp_t[..., 1] / (Y / 2)
    disp_norm[..., 2] = disp_t[..., 2] / (Z / 2)
    
    # Apply displacement
    deformed_grid = base_grid + disp_norm.unsqueeze(0)
    warped_t = F.grid_sample(
        vol_t, deformed_grid, mode='bilinear',
        padding_mode=padding_mode, align_corners=True
    )
    
    # Convert back
    warped = warped_t.squeeze().cpu().numpy()
    warped = warped * (vol_max - vol_min) + vol_min
    
    return warped.astype(original_dtype)


def create_3d_grid_image(
    control_point_displacements: NDArray[np.floating],
    control_point_positions: NDArray[np.floating],
    image_shape: tuple[int, int, int],
    axis: int = 0,
    point_value: int = 255,
    line_value: int = 128,
) -> NDArray[np.uint8]:
    """
    Create a MIP visualization of the 3D FFD control grid.
    
    Parameters
    ----------
    control_point_displacements : ndarray
        Control point displacements of shape (n_ctrl_z, n_ctrl_y, n_ctrl_x, 3).
    control_point_positions : ndarray
        Original control point positions of shape (n_ctrl_z, n_ctrl_y, n_ctrl_x, 3).
    image_shape : tuple
        Volume shape (Z, Y, X).
    axis : int
        Axis to project along for MIP visualization.
        0 = Z (front view), 1 = Y (top view), 2 = X (side view). Default 0.
    point_value : int
        Pixel value for control points. Default 255.
    line_value : int
        Pixel value for grid lines. Default 128.
    
    Returns
    -------
    grid_image : ndarray
        MIP visualization image with dtype uint8.
        Shape depends on axis: (Y, X), (Z, X), or (Z, Y).
    """
    Z, Y, X = image_shape
    n_ctrl_z, n_ctrl_y, n_ctrl_x = control_point_displacements.shape[:3]
    
    # Determine output shape based on axis
    if axis == 0:  # Project along Z
        out_shape = (Y, X)
        proj_dims = (1, 0)  # y, x
    elif axis == 1:  # Project along Y
        out_shape = (Z, X)
        proj_dims = (2, 0)  # z, x
    else:  # axis == 2, project along X
        out_shape = (Z, Y)
        proj_dims = (2, 1)  # z, y
    
    image = np.zeros(out_shape, dtype=np.uint8)
    
    # Compute deformed positions
    deformed_positions = control_point_positions + control_point_displacements
    
    # Extract 2D projections of all control points
    proj_positions = np.zeros((n_ctrl_z, n_ctrl_y, n_ctrl_x, 2), dtype=np.float32)
    for iz in range(n_ctrl_z):
        for iy in range(n_ctrl_y):
            for ix in range(n_ctrl_x):
                proj_positions[iz, iy, ix, 0] = deformed_positions[iz, iy, ix, proj_dims[1]]
                proj_positions[iz, iy, ix, 1] = deformed_positions[iz, iy, ix, proj_dims[0]]
    
    # Draw lines helper
    def draw_line(p1, p2, img, val):
        p1, p2 = np.array(p1), np.array(p2)
        diff = p2 - p1
        max_steps = int(np.ceil(np.linalg.norm(diff))) + 1
        for t in np.linspace(0, 1, max_steps):
            pt = p1 + t * diff
            c, r = int(round(pt[0])), int(round(pt[1]))
            if 0 <= r < out_shape[0] and 0 <= c < out_shape[1]:
                if img[r, c] != point_value:
                    img[r, c] = max(img[r, c], val)
    
    # Draw grid lines connecting adjacent control points
    for iz in range(n_ctrl_z):
        for iy in range(n_ctrl_y):
            for ix in range(n_ctrl_x):
                # Connect to x+1
                if ix < n_ctrl_x - 1:
                    draw_line(proj_positions[iz, iy, ix], proj_positions[iz, iy, ix + 1], image, line_value)
                # Connect to y+1
                if iy < n_ctrl_y - 1:
                    draw_line(proj_positions[iz, iy, ix], proj_positions[iz, iy + 1, ix], image, line_value)
                # Connect to z+1
                if iz < n_ctrl_z - 1:
                    draw_line(proj_positions[iz, iy, ix], proj_positions[iz + 1, iy, ix], image, line_value)
    
    # Draw control points (on top)
    for iz in range(n_ctrl_z):
        for iy in range(n_ctrl_y):
            for ix in range(n_ctrl_x):
                c, r = proj_positions[iz, iy, ix]
                ci, ri = int(round(c)), int(round(r))
                if 0 <= ri < out_shape[0] and 0 <= ci < out_shape[1]:
                    image[ri, ci] = point_value
    
    return image

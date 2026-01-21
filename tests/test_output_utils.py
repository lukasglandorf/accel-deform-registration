# -*- coding: utf-8 -*-
"""
Utilities for saving test outputs including images, volumes, and grids.

This module provides functions to:
- Save 2D images as JPEG
- Save 3D volumes as NIfTI (.nii.gz)
- Create deformable checkerboard patterns
- Draw control point grids with round nodes (centered)
- Compute control point position errors
"""

from __future__ import annotations

import os
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, Dict, Any


def ensure_output_dir(output_dir: str) -> str:
    """Create output directory if it doesn't exist."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    return output_dir


def normalize_to_uint8(img: NDArray) -> NDArray[np.uint8]:
    """Normalize an image to uint8 [0, 255] range."""
    img = np.asarray(img, dtype=np.float64)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def save_2d_image(
    img: NDArray,
    filepath: str,
    normalize: bool = True,
) -> str:
    """
    Save a 2D image as JPEG.
    
    Parameters
    ----------
    img : ndarray
        2D image array.
    filepath : str
        Output path (will add .jpg if not present).
    normalize : bool
        If True, normalize to [0, 255]. Default True.
    
    Returns
    -------
    filepath : str
        Actual path where file was saved.
    """
    from PIL import Image
    
    if not filepath.lower().endswith(('.jpg', '.jpeg')):
        filepath = filepath + '.jpg'
    
    ensure_output_dir(os.path.dirname(filepath))
    
    if normalize:
        img = normalize_to_uint8(img)
    else:
        img = np.asarray(img).astype(np.uint8)
    
    # Handle grayscale
    if img.ndim == 2:
        pil_img = Image.fromarray(img, mode='L')
    else:
        pil_img = Image.fromarray(img)
    
    pil_img.save(filepath, quality=95)
    return filepath


def save_3d_volume(
    vol: NDArray,
    filepath: str,
    affine: Optional[NDArray] = None,
) -> str:
    """
    Save a 3D volume as NIfTI (.nii.gz).
    
    Parameters
    ----------
    vol : ndarray
        3D volume array (Z, Y, X).
    filepath : str
        Output path (will add .nii.gz if not present).
    affine : ndarray, optional
        4x4 affine matrix. If None, uses identity.
    
    Returns
    -------
    filepath : str
        Actual path where file was saved.
    """
    import nibabel as nib
    
    if not filepath.lower().endswith('.nii.gz'):
        if filepath.lower().endswith('.nii'):
            filepath = filepath + '.gz'
        else:
            filepath = filepath + '.nii.gz'
    
    ensure_output_dir(os.path.dirname(filepath))
    
    if affine is None:
        affine = np.eye(4)
    
    # Convert to float32 for NIfTI
    vol = np.asarray(vol, dtype=np.float32)
    
    nii = nib.Nifti1Image(vol, affine)
    nib.save(nii, filepath)
    return filepath


def draw_circle(
    img: NDArray[np.uint8],
    center: Tuple[int, int],
    radius: int,
    value: int,
) -> None:
    """Draw a filled circle on an image (in-place)."""
    Y, X = img.shape[:2]
    cy, cx = center
    
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx*dx + dy*dy <= radius*radius:
                y, x = cy + dy, cx + dx
                if 0 <= y < Y and 0 <= x < X:
                    img[y, x] = value


# ============================================================================
# Centered Control Point Grid Functions
# ============================================================================

def compute_centered_grid_positions_2d(
    image_shape: Tuple[int, int],
    grid_spacing: int,
    boundary_layers: int = 1,
) -> NDArray[np.float32]:
    """
    Compute centered control point positions for 2D images.
    
    The grid is centered such that control points span from the image center
    outward, with optional boundary layers outside the image.
    
    Parameters
    ----------
    image_shape : tuple
        Image shape (Y, X).
    grid_spacing : int
        Spacing between control points in pixels.
    boundary_layers : int
        Number of control point rows outside the image boundary. Default 1.
    
    Returns
    -------
    positions : ndarray
        Control point positions of shape (n_ctrl_y, n_ctrl_x, 2).
        The last dimension is (x, y) coordinates.
    """
    Y, X = image_shape
    
    # Compute number of control points needed to cover the image
    # We want the grid centered on the image
    n_inside_x = int(np.ceil(X / grid_spacing)) + 1
    n_inside_y = int(np.ceil(Y / grid_spacing)) + 1
    
    # Total with boundary layers on each side
    n_ctrl_x = n_inside_x + 2 * boundary_layers
    n_ctrl_y = n_inside_y + 2 * boundary_layers
    
    # Compute the start position so grid is centered
    # The interior grid spans [0, X] and [0, Y] approximately
    total_span_x = (n_inside_x - 1) * grid_spacing
    total_span_y = (n_inside_y - 1) * grid_spacing
    
    start_x = (X - total_span_x) / 2 - boundary_layers * grid_spacing
    start_y = (Y - total_span_y) / 2 - boundary_layers * grid_spacing
    
    positions = np.zeros((n_ctrl_y, n_ctrl_x, 2), dtype=np.float32)
    
    for iy in range(n_ctrl_y):
        for ix in range(n_ctrl_x):
            positions[iy, ix, 0] = start_x + ix * grid_spacing  # x
            positions[iy, ix, 1] = start_y + iy * grid_spacing  # y
    
    return positions


def compute_centered_grid_positions_3d(
    image_shape: Tuple[int, int, int],
    grid_spacing: int,
    boundary_layers: int = 1,
) -> NDArray[np.float32]:
    """
    Compute centered control point positions for 3D volumes.
    
    Parameters
    ----------
    image_shape : tuple
        Volume shape (Z, Y, X).
    grid_spacing : int
        Spacing between control points in voxels.
    boundary_layers : int
        Number of control point layers outside the volume boundary. Default 1.
    
    Returns
    -------
    positions : ndarray
        Control point positions of shape (n_ctrl_z, n_ctrl_y, n_ctrl_x, 3).
        The last dimension is (x, y, z) coordinates.
    """
    Z, Y, X = image_shape
    
    n_inside_x = int(np.ceil(X / grid_spacing)) + 1
    n_inside_y = int(np.ceil(Y / grid_spacing)) + 1
    n_inside_z = int(np.ceil(Z / grid_spacing)) + 1
    
    n_ctrl_x = n_inside_x + 2 * boundary_layers
    n_ctrl_y = n_inside_y + 2 * boundary_layers
    n_ctrl_z = n_inside_z + 2 * boundary_layers
    
    total_span_x = (n_inside_x - 1) * grid_spacing
    total_span_y = (n_inside_y - 1) * grid_spacing
    total_span_z = (n_inside_z - 1) * grid_spacing
    
    start_x = (X - total_span_x) / 2 - boundary_layers * grid_spacing
    start_y = (Y - total_span_y) / 2 - boundary_layers * grid_spacing
    start_z = (Z - total_span_z) / 2 - boundary_layers * grid_spacing
    
    positions = np.zeros((n_ctrl_z, n_ctrl_y, n_ctrl_x, 3), dtype=np.float32)
    
    for iz in range(n_ctrl_z):
        for iy in range(n_ctrl_y):
            for ix in range(n_ctrl_x):
                positions[iz, iy, ix, 0] = start_x + ix * grid_spacing  # x
                positions[iz, iy, ix, 1] = start_y + iy * grid_spacing  # y
                positions[iz, iy, ix, 2] = start_z + iz * grid_spacing  # z
    
    return positions


def create_2d_grid_image_with_circles(
    control_point_displacements: NDArray[np.floating],
    control_point_positions: NDArray[np.floating],
    image_shape: Tuple[int, int],
    point_radius: int = 4,
    point_value: int = 255,
    line_value: int = 128,
    line_thickness: int = 2,
) -> NDArray[np.uint8]:
    """
    Create a visualization of the 2D FFD control grid with circular nodes.
    
    Parameters
    ----------
    control_point_displacements : ndarray
        Control point displacements of shape (n_ctrl_y, n_ctrl_x, 2).
    control_point_positions : ndarray
        Original control point positions of shape (n_ctrl_y, n_ctrl_x, 2).
    image_shape : tuple
        Output image shape (Y, X).
    point_radius : int
        Radius of control point circles in pixels. Default 4.
    point_value : int
        Pixel value for control points. Default 255.
    line_value : int
        Pixel value for grid lines. Default 128.
    line_thickness : int
        Thickness of grid lines in pixels. Default 2.
    
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
    
    def draw_line(p1, p2, img, val, thickness):
        p1, p2 = np.array(p1), np.array(p2)
        diff = p2 - p1
        length = np.linalg.norm(diff)
        if length < 1:
            return
        
        max_steps = int(np.ceil(length)) * 2 + 1
        
        for t in np.linspace(0, 1, max_steps):
            pt = p1 + t * diff
            x, y = int(round(pt[0])), int(round(pt[1]))
            
            for dy in range(-thickness//2, thickness//2 + 1):
                for dx in range(-thickness//2, thickness//2 + 1):
                    xi, yi = x + dx, y + dy
                    if 0 <= xi < X and 0 <= yi < Y:
                        if img[yi, xi] < point_value:
                            img[yi, xi] = max(img[yi, xi], val)
    
    # Draw horizontal lines
    for iy in range(n_ctrl_y):
        for ix in range(n_ctrl_x - 1):
            draw_line(
                deformed_positions[iy, ix],
                deformed_positions[iy, ix + 1],
                image, line_value, line_thickness
            )
    
    # Draw vertical lines
    for iy in range(n_ctrl_y - 1):
        for ix in range(n_ctrl_x):
            draw_line(
                deformed_positions[iy, ix],
                deformed_positions[iy + 1, ix],
                image, line_value, line_thickness
            )
    
    # Draw control points as circles (on top)
    for iy in range(n_ctrl_y):
        for ix in range(n_ctrl_x):
            x, y = deformed_positions[iy, ix]
            xi, yi = int(round(x)), int(round(y))
            draw_circle(image, (yi, xi), point_radius, point_value)
    
    return image


def create_3d_grid_image_with_circles(
    control_point_displacements: NDArray[np.floating],
    control_point_positions: NDArray[np.floating],
    image_shape: Tuple[int, int, int],
    axis: int = 0,
    point_radius: int = 3,
    point_value: int = 255,
    line_value: int = 128,
    line_thickness: int = 2,
) -> NDArray[np.uint8]:
    """
    Create a MIP visualization of the 3D FFD control grid with circular nodes.
    
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
    point_radius : int
        Radius of control point circles in pixels. Default 3.
    point_value : int
        Pixel value for control points. Default 255.
    line_value : int
        Pixel value for grid lines. Default 128.
    line_thickness : int
        Thickness of grid lines in pixels. Default 2.
    
    Returns
    -------
    grid_image : ndarray
        MIP visualization image with dtype uint8.
    """
    Z, Y, X = image_shape
    n_ctrl_z, n_ctrl_y, n_ctrl_x = control_point_displacements.shape[:3]
    
    if axis == 0:  # Project along Z
        out_shape = (Y, X)
        proj_dims = (1, 0)
    elif axis == 1:  # Project along Y
        out_shape = (Z, X)
        proj_dims = (2, 0)
    else:  # axis == 2, project along X
        out_shape = (Z, Y)
        proj_dims = (2, 1)
    
    image = np.zeros(out_shape, dtype=np.uint8)
    
    deformed_positions = control_point_positions + control_point_displacements
    
    proj_positions = np.zeros((n_ctrl_z, n_ctrl_y, n_ctrl_x, 2), dtype=np.float32)
    for iz in range(n_ctrl_z):
        for iy in range(n_ctrl_y):
            for ix in range(n_ctrl_x):
                proj_positions[iz, iy, ix, 0] = deformed_positions[iz, iy, ix, proj_dims[1]]
                proj_positions[iz, iy, ix, 1] = deformed_positions[iz, iy, ix, proj_dims[0]]
    
    def draw_line(p1, p2, img, val, thickness):
        p1, p2 = np.array(p1), np.array(p2)
        diff = p2 - p1
        length = np.linalg.norm(diff)
        if length < 1:
            return
        
        max_steps = int(np.ceil(length)) * 2 + 1
        
        for t in np.linspace(0, 1, max_steps):
            pt = p1 + t * diff
            c, r = int(round(pt[0])), int(round(pt[1]))
            
            for dy in range(-thickness//2, thickness//2 + 1):
                for dx in range(-thickness//2, thickness//2 + 1):
                    ci, ri = c + dx, r + dy
                    if 0 <= ri < out_shape[0] and 0 <= ci < out_shape[1]:
                        if img[ri, ci] < point_value:
                            img[ri, ci] = max(img[ri, ci], val)
    
    for iz in range(n_ctrl_z):
        for iy in range(n_ctrl_y):
            for ix in range(n_ctrl_x):
                if ix < n_ctrl_x - 1:
                    draw_line(proj_positions[iz, iy, ix], 
                             proj_positions[iz, iy, ix + 1], image, line_value, line_thickness)
                if iy < n_ctrl_y - 1:
                    draw_line(proj_positions[iz, iy, ix], 
                             proj_positions[iz, iy + 1, ix], image, line_value, line_thickness)
                if iz < n_ctrl_z - 1:
                    draw_line(proj_positions[iz, iy, ix], 
                             proj_positions[iz + 1, iy, ix], image, line_value, line_thickness)
    
    for iz in range(n_ctrl_z):
        for iy in range(n_ctrl_y):
            for ix in range(n_ctrl_x):
                c, r = proj_positions[iz, iy, ix]
                ci, ri = int(round(c)), int(round(r))
                draw_circle(image, (ri, ci), point_radius, point_value)
    
    return image


# ============================================================================
# Deformable Checkerboard Functions
# ============================================================================

def create_checkerboard_pattern_2d(
    image_shape: Tuple[int, int],
    grid_spacing: int,
    boundary_layers: int = 1,
    light_value: int = 200,
    dark_value: int = 80,
) -> NDArray[np.uint8]:
    """
    Create a 2D checkerboard pattern aligned with control point grid.
    
    The checkerboard squares have the same corner coordinates as control points.
    
    Parameters
    ----------
    image_shape : tuple
        Image shape (Y, X).
    grid_spacing : int
        Spacing between control points (size of checkerboard squares).
    boundary_layers : int
        Number of control point rows outside boundary. Default 1.
    light_value : int
        Gray value for light squares. Default 200.
    dark_value : int
        Gray value for dark squares. Default 80.
    
    Returns
    -------
    checkerboard : ndarray
        Checkerboard pattern of shape (Y, X) with dtype uint8.
    """
    Y, X = image_shape
    
    # Get the same positions as the control points
    ctrl_pos = compute_centered_grid_positions_2d(image_shape, grid_spacing, boundary_layers)
    
    # Starting point of the first control point
    start_x = ctrl_pos[0, 0, 0]
    start_y = ctrl_pos[0, 0, 1]
    
    checkerboard = np.zeros((Y, X), dtype=np.uint8)
    
    for y in range(Y):
        for x in range(X):
            # Which cell are we in?
            ix = int(np.floor((x - start_x) / grid_spacing))
            iy = int(np.floor((y - start_y) / grid_spacing))
            
            if (ix + iy) % 2 == 0:
                checkerboard[y, x] = light_value
            else:
                checkerboard[y, x] = dark_value
    
    return checkerboard


def create_checkerboard_pattern_3d(
    volume_shape: Tuple[int, int, int],
    grid_spacing: int,
    boundary_layers: int = 1,
    light_value: int = 200,
    dark_value: int = 80,
) -> NDArray[np.uint8]:
    """
    Create a 3D checkerboard pattern aligned with control point grid.
    
    Parameters
    ----------
    volume_shape : tuple
        Volume shape (Z, Y, X).
    grid_spacing : int
        Spacing between control points (size of checkerboard cubes).
    boundary_layers : int
        Number of control point layers outside boundary. Default 1.
    light_value : int
        Gray value for light cubes. Default 200.
    dark_value : int
        Gray value for dark cubes. Default 80.
    
    Returns
    -------
    checkerboard : ndarray
        Checkerboard pattern of shape (Z, Y, X) with dtype uint8.
    """
    Z, Y, X = volume_shape
    
    ctrl_pos = compute_centered_grid_positions_3d(volume_shape, grid_spacing, boundary_layers)
    
    start_x = ctrl_pos[0, 0, 0, 0]
    start_y = ctrl_pos[0, 0, 0, 1]
    start_z = ctrl_pos[0, 0, 0, 2]
    
    checkerboard = np.zeros((Z, Y, X), dtype=np.uint8)
    
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                ix = int(np.floor((x - start_x) / grid_spacing))
                iy = int(np.floor((y - start_y) / grid_spacing))
                iz = int(np.floor((z - start_z) / grid_spacing))
                
                if (ix + iy + iz) % 2 == 0:
                    checkerboard[z, y, x] = light_value
                else:
                    checkerboard[z, y, x] = dark_value
    
    return checkerboard


def apply_displacement_to_image_2d(
    image: NDArray,
    displacement_field: NDArray[np.floating],
) -> NDArray:
    """
    Apply a 2D displacement field to an image.
    
    Parameters
    ----------
    image : ndarray
        Input image of shape (Y, X).
    displacement_field : ndarray
        Displacement field of shape (Y, X, 2).
    
    Returns
    -------
    warped : ndarray
        Warped image.
    """
    import torch
    import torch.nn.functional as F
    
    Y, X = image.shape
    device = torch.device('cpu')
    
    # Convert to torch tensors
    img_tensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    disp_tensor = torch.from_numpy(displacement_field.astype(np.float32)).unsqueeze(0).to(device)
    
    # Create sampling grid
    yy, xx = torch.meshgrid(
        torch.arange(Y, dtype=torch.float32, device=device),
        torch.arange(X, dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    # Apply displacement
    sample_x = xx - disp_tensor[0, ..., 0]
    sample_y = yy - disp_tensor[0, ..., 1]
    
    # Normalize to [-1, 1]
    sample_x = 2.0 * sample_x / (X - 1) - 1.0
    sample_y = 2.0 * sample_y / (Y - 1) - 1.0
    
    grid = torch.stack([sample_x, sample_y], dim=-1).unsqueeze(0)
    
    warped = F.grid_sample(img_tensor, grid, mode='bilinear', padding_mode='border', align_corners=True)
    
    return warped.squeeze().numpy()


def apply_displacement_to_volume_3d(
    volume: NDArray,
    displacement_field: NDArray[np.floating],
) -> NDArray:
    """
    Apply a 3D displacement field to a volume.
    
    Parameters
    ----------
    volume : ndarray
        Input volume of shape (Z, Y, X).
    displacement_field : ndarray
        Displacement field of shape (Z, Y, X, 3).
    
    Returns
    -------
    warped : ndarray
        Warped volume.
    """
    import torch
    import torch.nn.functional as F
    
    Z, Y, X = volume.shape
    device = torch.device('cpu')
    
    vol_tensor = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    disp_tensor = torch.from_numpy(displacement_field.astype(np.float32)).unsqueeze(0).to(device)
    
    zz, yy, xx = torch.meshgrid(
        torch.arange(Z, dtype=torch.float32, device=device),
        torch.arange(Y, dtype=torch.float32, device=device),
        torch.arange(X, dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    sample_x = xx - disp_tensor[0, ..., 0]
    sample_y = yy - disp_tensor[0, ..., 1]
    sample_z = zz - disp_tensor[0, ..., 2]
    
    sample_x = 2.0 * sample_x / (X - 1) - 1.0
    sample_y = 2.0 * sample_y / (Y - 1) - 1.0
    sample_z = 2.0 * sample_z / (Z - 1) - 1.0
    
    grid = torch.stack([sample_x, sample_y, sample_z], dim=-1).unsqueeze(0)
    
    # Note: For 5D inputs, grid_sample uses 'bilinear' mode (which does trilinear interpolation)
    warped = F.grid_sample(vol_tensor, grid, mode='bilinear', padding_mode='border', align_corners=True)
    
    return warped.squeeze().numpy()


# ============================================================================
# Control Point Error Metrics
# ============================================================================

def compute_control_point_error(
    true_positions: NDArray[np.floating],
    estimated_positions: NDArray[np.floating],
    method: str = 'rmse',
) -> Dict[str, float]:
    """
    Compute error between true and estimated control point positions.
    
    Parameters
    ----------
    true_positions : ndarray
        True deformed control point positions, shape (..., ndim).
    estimated_positions : ndarray
        Estimated (registered) control point positions, shape (..., ndim).
    method : str
        Error method: 'rmse' (root mean squared), 'mae' (mean absolute),
        or 'both'. Default 'rmse'.
    
    Returns
    -------
    errors : dict
        Dictionary with error metrics:
        - 'rmse': Root mean squared Euclidean distance
        - 'mae': Mean absolute Euclidean distance
        - 'max': Maximum Euclidean distance
        - 'std': Standard deviation of Euclidean distances
    """
    # Compute Euclidean distances
    diff = true_positions - estimated_positions
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))
    
    errors = {
        'rmse': np.sqrt(np.mean(distances ** 2)),
        'mae': np.mean(distances),
        'max': np.max(distances),
        'std': np.std(distances),
        'median': np.median(distances),
    }
    
    return errors


def compute_displacement_field_error(
    true_displacement: NDArray[np.floating],
    estimated_displacement: NDArray[np.floating],
) -> Dict[str, float]:
    """
    Compute error between true and estimated displacement fields.
    
    Parameters
    ----------
    true_displacement : ndarray
        True displacement field, shape (..., ndim).
    estimated_displacement : ndarray
        Estimated displacement field, shape (..., ndim).
    
    Returns
    -------
    errors : dict
        Dictionary with error metrics.
    """
    diff = true_displacement - estimated_displacement
    magnitudes = np.sqrt(np.sum(diff ** 2, axis=-1))
    
    return {
        'rmse': np.sqrt(np.mean(magnitudes ** 2)),
        'mae': np.mean(magnitudes),
        'max': np.max(magnitudes),
        'std': np.std(magnitudes),
    }


# ============================================================================
# OCTA Data Loading
# ============================================================================

def get_octa_data_dir() -> str:
    """Get the path to the OCTA test data directory."""
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "data", "octa")


def check_octa_data_available() -> Tuple[bool, Dict[str, Any]]:
    """
    Check if OCTA test data is available.
    
    Expected files in tests/data/octa/:
    - target_2d.tif or target_2d.png: 2D MIP target image
    - moving_2d.tif or moving_2d.png: 2D MIP moving image
    - target_3d.nii.gz or target_3d.tif: 3D target volume
    - moving_3d.nii.gz or moving_3d.tif: 3D moving volume
    
    Returns
    -------
    available : bool
        True if any OCTA data files are found.
    files : dict
        Dictionary of available files with keys:
        - 'target_2d': path to 2D target
        - 'moving_2d': path to 2D moving
        - 'target_3d': path to 3D target
        - 'moving_3d': path to 3D moving
    """
    octa_dir = get_octa_data_dir()
    
    files = {
        'target_2d': None,
        'moving_2d': None,
        'target_3d': None,
        'moving_3d': None,
    }
    
    if not os.path.exists(octa_dir):
        return False, files
    
    # Check for 2D files
    for ext in ['.tif', '.tiff', '.png', '.jpg']:
        path_t = os.path.join(octa_dir, f'target_2d{ext}')
        path_m = os.path.join(octa_dir, f'moving_2d{ext}')
        if os.path.exists(path_t):
            files['target_2d'] = path_t
        if os.path.exists(path_m):
            files['moving_2d'] = path_m
    
    # Check for 3D files
    for ext in ['.nii.gz', '.nii', '.tif', '.tiff']:
        path_t = os.path.join(octa_dir, f'target_3d{ext}')
        path_m = os.path.join(octa_dir, f'moving_3d{ext}')
        if os.path.exists(path_t):
            files['target_3d'] = path_t
        if os.path.exists(path_m):
            files['moving_3d'] = path_m
    
    available = any(v is not None for v in files.values())
    return available, files


def load_octa_image(filepath: str) -> NDArray[np.float32]:
    """
    Load an OCTA image file.
    
    Supports TIFF, PNG, JPEG, and NIfTI formats.
    
    Parameters
    ----------
    filepath : str
        Path to the image file.
    
    Returns
    -------
    image : ndarray
        Image array normalized to [0, 1].
    """
    import os
    ext = os.path.splitext(filepath.lower())[1]
    
    if ext in ['.gz']:
        ext = '.nii.gz' if filepath.lower().endswith('.nii.gz') else ext
    
    if ext in ['.nii', '.nii.gz']:
        import nibabel as nib
        img = nib.load(filepath).get_fdata()
    elif ext in ['.tif', '.tiff']:
        import tifffile
        img = tifffile.imread(filepath)
    else:  # PNG, JPG, etc.
        from PIL import Image
        img = np.array(Image.open(filepath))
    
    # Convert to float and normalize
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / img.max()
    
    return img


# ============================================================================
# Statistics Helpers
# ============================================================================

def compute_stats(values: list) -> Dict[str, float]:
    """Compute mean and std of a list of values."""
    values = np.array(values)
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
    }
